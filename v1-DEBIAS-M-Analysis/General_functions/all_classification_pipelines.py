## contains functions that take as input the raw and transformed data
## and runs the predictive pipelines

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import SBCMP
from debiasm import DebiasMClassifier
import torch
from General_functions.delong import delong_roc_test
from scipy.stats import combine_pvalues
from debiasm.torch_functions import rescale
from skbio.stats.composition import clr

eps=1e-10


def sbc_transform(X, self):
    x = torch.tensor(X.values)
    batch_inds, x = x[:, 0], x[:, 1:]
    x = x + self.batch_weights[batch_inds.long()]
    x=F.normalize(x, p=1)
    return( pd.DataFrame(x.detach().numpy(), 
                         index=X.index, 
                         columns=X.columns[1:]))


def sbc_multiplicative_transform(X, self):
    x = torch.tensor(X.values)
    batch_inds, x = x[:, 0], x[:, 1:]
    x = F.normalize( torch.pow(2, self.batch_weights[batch_inds.long()] ) * x, p=1 )
    return( pd.DataFrame(x.detach().numpy(), 
                         index=X.index, 
                         columns=X.columns[1:]))

def run_predictions(y, 
                    df_with_batch, 
                    df_conqur, 
                    df_combat, 
                    seed=123, 
                    do_clr_transform=False,
                    b_str=1,
                    w_l2=0, 
                    l2_str = 0, 
                    learning_rate=0.005,
                    min_epochs=25,
                    df_snm=None, 
                    df_mup=None, 
                    df_pls=None, 
                    df_perc=None
                    ):
    print(df_with_batch.shape)
    data_relabund=df_with_batch.copy()
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if do_clr_transform:
        
        
        df_with_batch.iloc[:, 1:] += 1
        df_with_batch.iloc[:, 1:]  = df_with_batch.iloc[:, 1:].div( df_with_batch.iloc[:, 1:] .sum(axis=1), axis=0)
        
        df_with_batch.iloc[:, 1:] = clr( df_with_batch.iloc[:, 1:] )
        
        
        df_conqur += 1 
        df_conqur = df_conqur.div( df_conqur.sum(axis=1), axis=0)
        
        df_conqur = pd.DataFrame( clr(df_conqur), 
                                 index=df_conqur.index, 
                                 columns=df_conqur.columns )
        
        df_combat += 1 
        df_combat = df_combat.div( df_combat.sum(axis=1), axis=0)
        
        df_combat = pd.DataFrame( clr( df_combat), 
                                 index=df_combat.index, 
                                 columns=df_combat.columns )
        
        if df_pls is not None: # we assume that df_pls is already given in clr space here
            
            df_pls=df_pls  ## no need for pseudocounts, assume this happened before plsda input
            
        if df_mup is not None:
            
            if df_mup.sum(axis=1).mean() > 5:
                df_mup += 1 ## only add 1 pseudocount if we're in count space
            else:
                df_mup += 1e-6 
            df_mup = df_mup.div( df_mup.sum(axis=1), axis=0)

            df_mup = pd.DataFrame( clr( df_mup), 
                                    index=df_mup.index, 
                                     columns=df_mup.columns )
            
        if df_perc is not None:
            if df_perc.sum(axis=1).mean() > 5:
                df_perc+=1  ## only add 1 pseudocount if we're in count space
            else:
                df_perc += 1e-6
            df_perc = df_perc.div( df_perc.sum(axis=1), axis=0)

            df_perc = pd.DataFrame( clr( df_perc), 
                                    index=df_perc.index, 
                                     columns=df_perc.columns )

        
    
    if df_snm is not None:
        results_dict = {'models':{'linear':[], 
                                  'debias-m':[], 
                                  'combat':[], 
                                  'conqur':[],
                                  'snm':[]
                                 }, 
                        'aurocs':{'linear':[], 
                                  'debias-m':[], 
                                  'combat':[], 
                                  'conqur':[],
                                  'snm':[]
                                  }, 
                         'auprs':{'linear':[], 
                                  'debias-m':[], 
                                  'combat':[], 
                                  'conqur':[],
                                  'snm':[]
                                  }, 
                         'p-vs-debiasm':{'linear':[], 
                                  'debias-m':[], 
                                  'combat':[], 
                                  'conqur':[],
                                  'snm':[]
                                  }
                       }
    else:
        results_dict = {'models':{'linear':[], 
                                  'debias-m':[], 
                                  'combat':[], 
                                  'conqur':[]
                                 }, 
                        'aurocs':{'linear':[], 
                                  'debias-m':[], 
                                  'combat':[], 
                                  'conqur':[]
                                  },
                        'auprs':{'linear':[], 
                                  'debias-m':[], 
                                  'combat':[], 
                                  'conqur':[]
                                  },
                        'p-vs-debiasm':{'linear':[], 
                                  'debias-m':[], 
                                  'combat':[], 
                                  'conqur':[]
                                  }
                       }
        
    if df_mup is not None:
        results_dict['models']['mmuphin']=[]
        results_dict['aurocs']['mmuphin']=[]
        results_dict['auprs']['mmuphin']=[]
        results_dict['p-vs-debiasm']['mmuphin']=[]
        
    if df_pls is not None:
        results_dict['models']['plsda']=[]
        results_dict['aurocs']['plsda']=[]
        results_dict['auprs']['plsda']=[]
        results_dict['p-vs-debiasm']['plsda']=[]
        
    if df_perc is not None:
        results_dict['models']['percnorm']=[]
        results_dict['aurocs']['percnorm']=[]
        results_dict['auprs']['percnorm']=[]
        results_dict['p-vs-debiasm']['percnorm']=[]
        
    
        
        
    if do_clr_transform:
        results_dict['clr_datasets'] = {'raw':df_with_batch, 
                                        'conqur':df_conqur,
                                        'combat':df_combat, 
                                        'mmuphin':df_mup, 
                                        'percnorm':df_perc
                                       }
    
    if do_clr_transform:
        results_dict['models']['clr_of_debias']=[]
        results_dict['aurocs']['clr_of_debias']=[]
        results_dict['auprs']['clr_of_debias']=[]
        results_dict['p-vs-debiasm']['clr_of_debias']=[]
        clr_of_debias_rocs=[]
        clr_of_debias_auprs=[]

    linear_rocs = []
    sbcmp_rocs = []
    conqur_rocs = []
    combat_rocs = []
    snm_rocs = []
    mmuphin_rocs=[]
    pls_rocs=[]
    percnorm_rocs=[]
    
    linear_auprs = []
    sbcmp_auprs = []
    conqur_auprs = []
    combat_auprs = []
    snm_auprs = []
    mmuphin_auprs=[]
    pls_auprs=[]
    percnorm_auprs=[]
    
    
    largest_batch_n = df_with_batch[0].value_counts().values[0]
    for batch in df_with_batch[0].unique():
        print(linear_rocs, sbcmp_rocs, conqur_rocs, 
              combat_rocs, snm_rocs,
              mmuphin_rocs, pls_rocs, percnorm_rocs)

        val_inds = df_with_batch[0]==batch
        X_train, X_val = df_with_batch[~val_inds], df_with_batch[val_inds]
        y_train, y_val = y[~val_inds], y[val_inds]


        if (np.unique( y_val ).shape[0]>1):# and (y_val.shape[0]<largest_batch_n):

            if not do_clr_transform:
                val_preds, mod = SBCMP.SBCMP_train_and_pred(X_train.values, 
                                                            X_val.values, 
                                                            y_train.values, 
                                                            y_val.values, 
                                                            batch_sim_strength=b_str, 
                                                            learning_rate=learning_rate,
                                                            test_split=0,
                                                            min_epochs=min_epochs,
                                                            w_l2 = w_l2, 
                                                            l2_strength=l2_str
                                                            )
                
                
                
            else:
            
                val_preds, mod =  SBCMP_train_and_pred_log_additive(X_train.values, 
                                                                    X_val.values, 
                                                                    y_train.values, 
                                                                    y_val.values, 
                                                                    batch_sim_strength=b_str,
                                                                    learning_rate=learning_rate,
                                                                    test_split=0,
                                                                    min_epochs=min_epochs,
                                                                    use_log=False,
                                                                    w_l2 = w_l2,
                                                                    l2_strength=l2_str
                                                                    )
                
                

            sbcmp_rocs.append(roc_auc_score(y_val, val_preds))
            sbcmp_auprs.append(average_precision_score(y_val, val_preds))
            
            results_dict['models']['debias-m'] += [mod]
            
            
            
            
            if not do_clr_transform:
                ss=StandardScaler()
                lr=LogisticRegression(penalty='none',
                                      max_iter=25000)
                

                lr.fit( ss.fit_transform( X_train.iloc[:, 1:].div(eps + X_train.iloc[:, 1:].sum(axis=1), 
                                                                 axis=0)
                                       ), y_train)
                linear_rocs.append( roc_auc_score(y_val, 
                                      lr.predict_proba( 
                                          ss.transform( X_val.iloc[:, 1:]\
                                                           .div(eps+X_val.iloc[:, 1:].sum(axis=1), 
                                                                axis=0) ) )[:, 1]
                                     ) )
                
                linear_auprs.append( average_precision_score(y_val, 
                                      lr.predict_proba( 
                                          ss.transform( X_val.iloc[:, 1:]\
                                                           .div(eps+X_val.iloc[:, 1:].sum(axis=1), 
                                                                axis=0) ) )[:, 1]
                                     ) )

                results_dict['models']['linear'] += [[ss,lr]]
                results_dict['p-vs-debiasm']['linear'] += [ np.power(10, 
                                                             delong_roc_test(y_val, 
                                                                             val_preds,
                                                    lr.predict_proba( 
                                          ss.transform( X_val.iloc[:, 1:]\
                                                           .div(eps+X_val.iloc[:, 1:].sum(axis=1), 
                                                                axis=0) ) )[:, 1]
                                                                            )) /2 ]
                
                if roc_auc_score(y_val, lr.predict_proba( 
                                          ss.transform( X_val.iloc[:, 1:]\
                                                           .div(eps+X_val.iloc[:, 1:].sum(axis=1), 
                                                                axis=0) ) )[:, 1]
                                    ) > \
                    roc_auc_score(y_val, val_preds):
                    
                    results_dict['p-vs-debiasm']['linear'][-1]=1-results_dict['p-vs-debiasm']['linear'][-1]
                
                


                ## conqur

                ss=StandardScaler()
                X_train, X_val = df_conqur[~val_inds], df_conqur[val_inds]
                lr=LogisticRegression(penalty='none',
                                      max_iter=25000)

                lr.fit(ss.fit_transform( X_train.div(eps + X_train.sum(axis=1),  axis=0) ), y_train)

                conqur_rocs.append( roc_auc_score(y_val, 
                                      lr.predict_proba( ss.transform( X_val\
                                                           .div(eps + X_val.sum(axis=1), 
                                                                axis=0) ) )[:, 1]
                                     ) )
                
                conqur_auprs.append( average_precision_score(y_val, 
                                      lr.predict_proba( ss.transform( X_val\
                                                           .div(eps + X_val.sum(axis=1), 
                                                                axis=0) ) )[:, 1]
                                     ) )


                results_dict['models']['conqur'] += [[ss,lr]]
                results_dict['p-vs-debiasm']['conqur'] += [ np.power(10, 
                                                             delong_roc_test(y_val, 
                                                                             val_preds,
                                                    lr.predict_proba( 
                                          ss.transform( X_val\
                                                           .div(eps+X_val.sum(axis=1), 
                                                                axis=0) ) )[:, 1]
                                                                            ))/2 ]
                
                if roc_auc_score(y_val, lr.predict_proba( 
                                          ss.transform( X_val\
                                                           .div(eps+X_val.sum(axis=1), 
                                                                axis=0) ) )[:, 1]
                                    ) > \
                    roc_auc_score(y_val, val_preds):
                    
                    results_dict['p-vs-debiasm']['conqur'][-1]=1-results_dict['p-vs-debiasm']['conqur'][-1]

                ## combat

                ss=StandardScaler()
                X_train, X_val = df_combat[~val_inds], df_combat[val_inds]
                lr=LogisticRegression(penalty='none',
                                      max_iter=25000)
                lr.fit( ss.fit_transform( X_train.div(eps + X_train.sum(axis=1), 
                                               axis=0) ), y_train)
                combat_rocs.append( roc_auc_score(y_val, 
                                      lr.predict_proba( ss.transform( X_val\
                                                           .div(eps+X_val.sum(axis=1), 
                                                                axis=0) ) )[:, 1]
                                     ) )
                
                combat_auprs.append( average_precision_score(y_val, 
                                      lr.predict_proba( ss.transform( X_val\
                                                           .div(eps+X_val.sum(axis=1), 
                                                                axis=0) ) )[:, 1]
                                     ) )
                
                

                results_dict['models']['combat'] += [[ss,lr]]
                results_dict['p-vs-debiasm']['combat'] += [ np.power(10, 
                                                             delong_roc_test(y_val, 
                                                                             val_preds,
                                                    lr.predict_proba( 
                                          ss.transform( X_val\
                                                           .div(eps+X_val.sum(axis=1), 
                                                                axis=0) ) )[:, 1]
                                                                            ))/2 ]
                
                if roc_auc_score(y_val, lr.predict_proba( 
                                          ss.transform( X_val\
                                                           .div(eps+X_val.sum(axis=1), 
                                                                axis=0) ) )[:, 1]
                                    ) > \
                    roc_auc_score(y_val, val_preds):
                    
                    results_dict['p-vs-debiasm']['combat'][-1]=1-results_dict['p-vs-debiasm']['combat'][-1]
                
                
                ## snm
                if df_snm is not None:
                    ss=StandardScaler()
                    X_train, X_val = df_snm[~val_inds], df_snm[val_inds]
                    lr=LogisticRegression(penalty='none',
                                          max_iter=25000)
                    
                    lr.fit( ss.fit_transform( X_train ), y_train)
                    snm_rocs.append( roc_auc_score(y_val, 
                                      lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                     ) )
                    
                    snm_auprs.append( average_precision_score(y_val, 
                                      lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                     ) )
                    
                    

                    results_dict['models']['snm'] +=  [[ss,lr]]
                    results_dict['p-vs-debiasm']['snm'] += [ np.power(10, 
                                                             delong_roc_test(y_val, 
                                                                             val_preds,
                                                 lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                                                            ))/2 ]
                
                if roc_auc_score(y_val, lr.predict_proba( ss.transform( X_val ) )[:, 1] 
                                    ) > \
                    roc_auc_score(y_val, val_preds):
                    
                    results_dict['p-vs-debiasm']['snm'][-1]=1-results_dict['p-vs-debiasm']['snm'][-1]
                        
                        
                ## mmuphin
                if df_mup is not None:
                    ss=StandardScaler()
                    X_train, X_val = rescale(df_mup[~val_inds]), rescale(df_mup[val_inds])
                    lr=LogisticRegression(penalty='none',
                                          max_iter=25000)
                    
                    lr.fit( ss.fit_transform( X_train ), y_train)
                    mmuphin_rocs.append( roc_auc_score(y_val, 
                                      lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                     ) )
                    
                    mmuphin_auprs.append( average_precision_score(y_val, 
                                      lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                     ) )

                    results_dict['models']['mmuphin'] +=  [[ss,lr]]
                    results_dict['p-vs-debiasm']['mmuphin'] += [ np.power(10, 
                                                             delong_roc_test(y_val, 
                                                                             val_preds,
                                                 lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                                                            ))/2 ]
                
                    if roc_auc_score(y_val, lr.predict_proba( ss.transform( X_val ) )[:, 1] 
                                        ) > \
                        roc_auc_score(y_val, val_preds):

                        results_dict['p-vs-debiasm']['mmuphin'][-1]=1-results_dict['p-vs-debiasm']['mmuphin'][-1]

                    
                ## plsda
                if df_pls is not None:
                    ss=StandardScaler()
                    X_train, X_val = df_pls[~val_inds], df_pls[val_inds]
                    lr=LogisticRegression(penalty='none',
                                          max_iter=25000)
                    
                    lr.fit( ss.fit_transform( X_train ), y_train)
                    pls_rocs.append( roc_auc_score(y_val, 
                                      lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                     ) )
                    
                    pls_auprs.append( average_precision_score(y_val, 
                                      lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                     ) )

                    results_dict['models']['plsda'] +=  [[ss,lr]]
                    results_dict['p-vs-debiasm']['plsda'] += [ np.power(10, 
                                                             delong_roc_test(y_val, 
                                                                             val_preds,
                                                 lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                                                            ))/2 ]
                
                    if roc_auc_score(y_val, lr.predict_proba( ss.transform( X_val ) )[:, 1] 
                                        ) > \
                        roc_auc_score(y_val, val_preds):

                        results_dict['p-vs-debiasm']['plsda'][-1]=1-results_dict['p-vs-debiasm']['plsda'][-1]
                        
                        
                ## percnorm
                if df_perc is not None:
                    ss=StandardScaler()
                    X_train, X_val = rescale(df_perc[~val_inds]), rescale(df_perc[val_inds])
                    lr=LogisticRegression(penalty='none',
                                          max_iter=25000)
                    
                    lr.fit( ss.fit_transform( X_train ), y_train)
                    percnorm_rocs.append( roc_auc_score(y_val, 
                                      lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                     ) )
                    
                    percnorm_auprs.append( average_precision_score(y_val, 
                                      lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                     ) )

                    results_dict['models']['percnorm'] +=  [[ss,lr]]
                    results_dict['p-vs-debiasm']['percnorm'] += [ np.power(10, 
                                                             delong_roc_test(y_val, 
                                                                             val_preds,
                                                 lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                                                            ))/2 ]
                
                    if roc_auc_score(y_val, lr.predict_proba( ss.transform( X_val ) )[:, 1] 
                                        ) > \
                        roc_auc_score(y_val, val_preds):

                        results_dict['p-vs-debiasm']['percnorm'][-1]=1-results_dict['p-vs-debiasm']['percnorm'][-1]
              
                
            else: # no relabund rescaling in clr data..
                lr=LogisticRegression(penalty='none',
                                      max_iter=25000)
                
                ss=StandardScaler()
                lr.fit( ss.fit_transform( X_train.iloc[:, 1:] ), y_train)
                linear_rocs.append( roc_auc_score(y_val, 
                                      lr.predict_proba( ss.transform( X_val.iloc[:, 1:] ) )[:, 1]
                                     ) )
                
                linear_auprs.append( average_precision_score(y_val, 
                                      lr.predict_proba( ss.transform( X_val.iloc[:, 1:] ) )[:, 1]
                                     ) )
                

                results_dict['models']['linear'] += [[ss,lr]]
                results_dict['p-vs-debiasm']['linear'] += [ np.power(10, 
                                                             delong_roc_test(y_val, 
                                                                             val_preds,
                                      lr.predict_proba( ss.transform( X_val.iloc[:, 1:] ) )[:, 1]
                                                                            ))/2 ]
                
                if roc_auc_score(y_val, lr.predict_proba( ss.transform( X_val.iloc[:, 1:] ) )[:, 1] 
                                    ) > \
                    roc_auc_score(y_val, val_preds):
                    
                    results_dict['p-vs-debiasm']['linear'][-1]=1-results_dict['p-vs-debiasm']['linear'][-1]
                        

                ## conqur

                ss=StandardScaler()
                X_train, X_val = df_conqur[~val_inds], df_conqur[val_inds]
                lr=LogisticRegression(penalty='none',
                                      max_iter=25000)

                lr.fit( ss.fit_transform( X_train ), y_train)

                conqur_rocs.append( roc_auc_score(y_val, 
                                      lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                     ) )
                
                conqur_auprs.append( average_precision_score(y_val, 
                                      lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                     ) )
                
                


                results_dict['models']['conqur'] += [[ss,lr]]
                
                ## one-tailed delong test
                results_dict['p-vs-debiasm']['conqur'] += [ np.power(10, 
                                                             delong_roc_test(y_val, 
                                                                             val_preds,
                                      lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                                                            ))/2 ]
                
                if roc_auc_score(y_val, lr.predict_proba( ss.transform( X_val ) )[:, 1] 
                                    ) > \
                    roc_auc_score(y_val, val_preds):
                    
                    results_dict['p-vs-debiasm']['conqur'][-1]=1-results_dict['p-vs-debiasm']['conqur'][-1]
                        
                
                
                
                ## combat
                ss=StandardScaler()
                X_train, X_val = df_combat[~val_inds], df_combat[val_inds]
                lr=LogisticRegression(penalty='none',
                                      max_iter=25000)
                lr.fit( ss.fit_transform( X_train ), y_train)
                combat_rocs.append( roc_auc_score(y_val, 
                                      lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                     ) )
                
                combat_auprs.append( average_precision_score(y_val, 
                                      lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                     ) )

                results_dict['models']['combat'] += [[ss,lr]]
                results_dict['p-vs-debiasm']['combat'] += [ np.power(10, 
                                                             delong_roc_test(y_val, 
                                                                             val_preds,
                                      lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                                                            ))/2 ]
                
                if roc_auc_score(y_val, lr.predict_proba( ss.transform( X_val ) )[:, 1] 
                                    ) > \
                    roc_auc_score(y_val, val_preds):
                    
                    results_dict['p-vs-debiasm']['combat'][-1]=1-results_dict['p-vs-debiasm']['combat'][-1]
                    
                    
                ## mmuphin
                if df_mup is not None:
                    ss=StandardScaler()
                    X_train, X_val = df_mup[~val_inds], df_mup[val_inds]
                    lr=LogisticRegression(penalty='none',
                                          max_iter=25000)
                    lr.fit( ss.fit_transform( X_train ), y_train)
                    mmuphin_rocs.append( roc_auc_score(y_val, 
                                          lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                         ) )
                    
                    mmuphin_auprs.append( average_precision_score(y_val, 
                                          lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                         ) )
                    
                    results_dict['models']['mmuphin'] += [[ss,lr]]
                    results_dict['p-vs-debiasm']['mmuphin'] += [ np.power(10, 
                                                                 delong_roc_test(y_val, 
                                                                                 val_preds,
                                          lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                                                                ))/2 ]

                    if roc_auc_score(y_val, lr.predict_proba( ss.transform( X_val ) )[:, 1] 
                                        ) > \
                        roc_auc_score(y_val, val_preds):

                        results_dict['p-vs-debiasm']['mmuphin'][-1]=1-results_dict['p-vs-debiasm']['mmuphin'][-1]
                        
                ## plsda
                if df_pls is not None:
                    ss=StandardScaler()
                    X_train, X_val = df_pls[~val_inds], df_pls[val_inds]
                    lr=LogisticRegression(penalty='none',
                                          max_iter=25000)
                    lr.fit( ss.fit_transform( X_train ), y_train)
                    pls_rocs.append( roc_auc_score(y_val, 
                                          lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                         ) )
                    
                    pls_auprs.append( average_precision_score(y_val, 
                                          lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                         ) )
                    
                    results_dict['models']['plsda'] += [[ss,lr]]
                    results_dict['p-vs-debiasm']['plsda'] += [ np.power(10, 
                                                                 delong_roc_test(y_val, 
                                                                                 val_preds,
                                          lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                                                                ))/2 ]

                    if roc_auc_score(y_val, lr.predict_proba( ss.transform( X_val ) )[:, 1] 
                                        ) > \
                        roc_auc_score(y_val, val_preds):

                        results_dict['p-vs-debiasm']['plsda'][-1]=1-results_dict['p-vs-debiasm']['plsda'][-1]
                        
                ## percnrom
                if df_perc is not None:
                    ss=StandardScaler()
                    X_train, X_val = df_perc[~val_inds], df_perc[val_inds]
                    lr=LogisticRegression(penalty='none',
                                          max_iter=25000)
                    lr.fit( ss.fit_transform( X_train ), y_train)
                    percnorm_rocs.append( roc_auc_score(y_val, 
                                          lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                         ) )
                    
                    percnorm_auprs.append( average_precision_score(y_val, 
                                          lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                         ) )

                    results_dict['models']['percnorm'] += [[ss,lr]]
                    results_dict['p-vs-debiasm']['percnorm'] += [ np.power(10, 
                                                                 delong_roc_test(y_val, 
                                                                                 val_preds,
                                          lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                                                                ))/2 ]

                    if roc_auc_score(y_val, lr.predict_proba( ss.transform( X_val ) )[:, 1] 
                                        ) > \
                        roc_auc_score(y_val, val_preds):

                        results_dict['p-vs-debiasm']['percnorm'][-1]=1-results_dict['p-vs-debiasm']['percnorm'][-1]
                
                 
                ### Running CLR( DEBIAS-M relabund output)
                
                X_train, X_val = data_relabund[~val_inds], data_relabund[val_inds]
                y_train, y_val = y[~val_inds], y[val_inds]

                val_preds, mod =  SBCMP_train_and_pred(X_train.values, 
                                                       X_val.values, 
                                                       y_train.values, 
                                                       y_val.values, 
                                                       batch_sim_strength=b_str*10,
                                                         #since the clr one is b_str/10
                                                       learning_rate=learning_rate,
                                                       test_split=0,
                                                       min_epochs=min_epochs,
                                                       w_l2 = w_l2,
                                                       l2_strength=l2_str
                                                       )
                ### run the transformation
                x=torch.Tensor(data_relabund.values)
                batch_inds, x = x[:, 0], x[:, 1:]
                x = F.normalize( torch.pow(2, mod.batch_weights[batch_inds.long()]
                                           ) * x, p=1 ).detach().numpy()
                
                clr_of_debias = pd.DataFrame( 
                           clr( rescale(1e-6 + x)),
                                index=data_relabund.index, 
                                columns=data_relabund.columns[1:]
                                   ) 
                
                ss=StandardScaler()
                X_train, X_val = clr_of_debias[~val_inds], clr_of_debias[val_inds]
                lr=LogisticRegression(penalty='none',
                                      max_iter=25000
                                      )
                
                lr.fit( ss.fit_transform( X_train ), y_train)
                clr_of_debias_rocs.append( roc_auc_score(y_val, 
                                      lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                     ) )

                clr_of_debias_auprs.append( average_precision_score(y_val, 
                                      lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                     ) )

                results_dict['models']['clr_of_debias'] += [[mod, ss,lr]]
                results_dict['p-vs-debiasm']['clr_of_debias'] += [ np.power(10, 
                                                             delong_roc_test(y_val, 
                                                                             val_preds,
                                      lr.predict_proba( ss.transform( X_val ) )[:, 1]
                                                                            ))/2 ]

                if roc_auc_score(y_val, lr.predict_proba( ss.transform( X_val ) )[:, 1] 
                                    ) > \
                    roc_auc_score(y_val, val_preds):

                    results_dict['p-vs-debiasm']['clr_of_debias'][-1]=1-results_dict['p-vs-debiasm']['clr_of_debias'][-1]
                
                
                    
                    
                    
                    
    results_dict['aurocs']['linear'] = linear_rocs
    results_dict['aurocs']['debias-m'] = sbcmp_rocs
    results_dict['aurocs']['combat'] = combat_rocs
    results_dict['aurocs']['conqur'] = conqur_rocs
    
    if df_snm is not None:
        results_dict['aurocs']['snm'] = snm_rocs
        
    if df_mup is not None:
        results_dict['aurocs']['mmuphin'] = mmuphin_rocs
        
    if df_pls is not None:
        results_dict['aurocs']['plsda'] = pls_rocs
        
    if df_perc is not None:
        results_dict['aurocs']['percnorm'] = percnorm_rocs
        
    if do_clr_transform:
        results_dict['aurocs']['clr_of_debias'] = clr_of_debias_rocs
        results_dict['auprs']['clr_of_debias'] = clr_of_debias_auprs
        
        
    results_dict['auprs']['linear'] = linear_auprs
    results_dict['auprs']['debias-m'] = sbcmp_auprs
    results_dict['auprs']['combat'] = combat_auprs
    results_dict['auprs']['conqur'] = conqur_auprs
    if df_snm is not None:
        results_dict['auprs']['snm'] = snm_auprs

    if df_mup is not None:
        results_dict['auprs']['mmuphin'] = mmuphin_auprs

    if df_pls is not None:
        results_dict['auprs']['plsda'] = pls_auprs

    if df_perc is not None:
        results_dict['auprs']['percnorm'] = percnorm_auprs
        
    pval_threshold = 1e-2
    results_dict['is_sig'] = {}
    for a in results_dict['p-vs-debiasm']:
        if a != 'debias-m':
            results_dict['is_sig'][a] = combine_pvalues( [b[0][0] 
                                                          for b in results_dict['p-vs-debiasm'][a]
                                                          if b == b ## nan check
                                                             ])[1] < pval_threshold 


    return(results_dict)





from SBCMP import *

class PL_SBCMP_log_additive(pl.LightningModule):
    """Logistic regression model."""

    def __init__(
        self,
        X, 
        batch_sim_strength: float,
        input_dim: int,
        num_classes: int,
        bias: bool = True,
        learning_rate: float = 1e-4,
        optimizer: Type[Optimizer] = Adam,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
        w_l2 : float = 0.0,
        use_log: bool=False,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            input_dim: number of dimensions of the input (at least 1)
            num_classes: number of class labels (binary: 2, multi-class: >2)
            bias: specifies if a constant or intercept should be fitted (equivalent to fit_intercept in sklearn)
            learning_rate: learning_rate for the optimizer
            optimizer: the optimizer to use (default: ``Adam``)
            l1_strength: L1 regularization strength (default: ``0.0``)
            l2_strength: L2 regularization strength (default: ``0.0``)
        """
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        
        
        self.linear = nn.Linear(in_features=self.hparams.input_dim, 
                                out_features=self.hparams.num_classes, 
                                bias=bias)

        self.X=X[:, 1:]
        self.bs=X[:, 0].long()
        self.unique_bs=self.bs.unique().long()
        self.n_batches=self.unique_bs.max()+1
        self.batch_weights = torch.nn.Parameter(data = torch.zeros(self.n_batches,
                                                                   input_dim))

        self.batch_sim_str=batch_sim_strength
        if use_log:
            self.processing_func = lambda x: torch.log(1e-1 + x)
        else:
            self.processing_func = lambda x: x
        
        
    def forward(self, x: Tensor) -> Tensor:
        batch_inds, x = x[:, 0], x[:, 1:]
        x = x + self.batch_weights[batch_inds.long()]
        x=F.normalize(x, p=1)
#         F.normalize( torch.pow(2, self.batch_weights[batch_inds.long()] ) * x, p=1 )
        x = self.linear(self.processing_func(x))
        y_hat = softmax(x)
        return y_hat

    
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch
        
        # flatten any input
        x = x.view(x.size(0), -1)

        y_hat = self.forward(x)
        
        
        # PyTorch cross_entropy function combines log_softmax and nll_loss in single function
        loss = F.cross_entropy(y_hat, y, reduction="sum")
        
        # L1 regularizer
        if self.hparams.l1_strength > 0:
            l1_reg = self.linear.weight.abs().sum()
            loss += self.hparams.l1_strength * l1_reg

        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = self.linear.weight.pow(2).sum()
            loss += self.hparams.l2_strength * l2_reg
            
        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = self.linear.weight.pow(2).sum()
            loss += self.hparams.l2_strength * l2_reg
            
        # DOMAIN similarity regularizer / bias correction
        if self.batch_sim_str > 0:
            x1 = torch.stack( [ ( self.batch_weights\
                                          [torch.where(self.unique_bs==a)[0]] + \
                    (self.X[ torch.where(self.bs==a)[0] ] )  \
                   ).mean(axis=0) for a in self.unique_bs ] )

            x1 = F.normalize(x1, p=1)

            loss += sum( [pairwise_distance(x1, a) for a in x1] ).sum() *\
                                    self.batch_sim_str
            
        # L2 regularizer for bias weight    
        if self.hparams.w_l2 > 0:
            # L2 regularizer for weighting parameter
            l2_reg = self.batch_weights.pow(2).sum()
            loss += self.hparams.w_l2 * l2_reg



        loss /= float( x.size(0) )
        return {"loss": loss}

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)
         # PyTorch cross_entropy function combines log_softmax and nll_loss in single function
        loss = F.cross_entropy(y_hat, y, reduction="sum")
        
        # L1 regularizer
        if self.hparams.l1_strength > 0:
            l1_reg = self.linear.weight.abs().sum()
            loss += self.hparams.l1_strength * l1_reg

        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = self.linear.weight.pow(2).sum()
            loss += self.hparams.l2_strength * l2_reg
        
        self.log('val_loss', loss)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        acc = 0
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        
        return {"val_loss": val_loss}

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)
        acc = 0#
        return {"test_loss": F.cross_entropy(y_hat, y),
                "acc": acc}

    def test_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        acc = 0
        test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        tensorboard_logs = {"test_ce_loss": test_loss, "test_acc": acc}
        progress_bar_metrics = tensorboard_logs
        return {"test_loss": test_loss,
                "log": tensorboard_logs,
                "progress_bar": progress_bar_metrics}

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        parser.add_argument("--input_dim", type=int, default=None)
        parser.add_argument("--num_classes", type=int, default=None)
        parser.add_argument("--bias", default="store_true")
        parser.add_argument("--batch_size", type=int, default=100)
        
        return parser
    
    
    
    
def SBCMP_train_and_pred_log_additive(X_train, 
                         X_val, 
                         y_train, 
                         y_val,
                         batch_sim_strength=1,
                         w_l2 = 0,
                         batch_size=None,
                         learning_rate=0.005,
                         l2_strength=0,
                         includes_batches=False,
                         val_split=0.1, 
                         test_split=0,
                         min_epochs=15, 
                         use_log=False
                         ):
    
    if batch_size is None:
        batch_size=X_train.shape[0]
    
    baseline_mod = LogisticRegression(max_iter=2500)
    baseline_mod.fit( X_train[:, 1:], y_train)
        
        
    model = PL_SBCMP_log_additive( X = torch.tensor( np.vstack((X_train, X_val)) ),
                      batch_sim_strength = batch_sim_strength,
                      input_dim = X_train.shape[1]-1, 
                      num_classes = 2, 
                      batch_size = batch_size,
                      learning_rate = learning_rate,
                      l2_strength = l2_strength, 
                      w_l2 = w_l2,
                      use_log=use_log
                    )
    
    ## initialize parameters to lbe similar to standard logistic regression
    model.linear.weight.data[0]=-torch.tensor( baseline_mod.coef_[0] )
    model.linear.weight.data[1]= torch.tensor( baseline_mod.coef_[0] )

    ## build pl dataloader
    dm = SklearnDataModule(X_train, 
                           y_train.astype(int),
                           val_split=val_split,
                           test_split=test_split
                           )

    ## run training
    trainer = pl.Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=2)], 
                             check_val_every_n_epoch=2, 
                             weights_summary=None, 
                             progress_bar_refresh_rate=0, 
                             min_epochs=min_epochs
                            )
    trainer.fit(model, 
                train_dataloaders=dm.train_dataloader(), 
                val_dataloaders=dm.val_dataloader()
               )
    
    ## get val predictions
    val_preds = model.forward( torch.tensor( X_val ).float() )[:, 1].detach().numpy()
    
    ## return predictions and the model
    return( val_preds, model )







