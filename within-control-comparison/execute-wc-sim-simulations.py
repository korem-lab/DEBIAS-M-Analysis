

import numpy as np
import pandas as pd
import sys
sys.path.append('../v1-DEBIAS-M-Analysis/')
from General_functions import data_loading, plotting
from cvc_debiasm import AdaptationDebiasMClassifierWithinControlSimilarity
from debiasm import AdaptationDebiasMClassifier

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
import torch
from scipy.stats import wilcoxon
from General_functions.plotting import global_palette
import matplotlib.pyplot as plt
import seaborn as sns
import os


def run_wc_sim_simulations(results_path='../results/within-control-comparison/'):

    df, md = data_loading.load_CRC('../data/CRC/')
    df_with_batch = pd.concat([pd.Series( pd.Categorical(md.Study).codes, 
                                             index=md.index), df], axis=1)    
    y=md.label

    logo = LeaveOneGroupOut()
    all_base_rocs=[]
    all_conly_rocs=[]
    seed=0
    for train_inds, test_inds in logo.split(df, md, md.Study):
        try:

            np.random.seed(seed)
            torch.manual_seed(seed)

            admc=AdaptationDebiasMClassifier()
            admc.fit(df_with_batch.loc[df.index[train_inds]].values,
                     md.loc[md.index[train_inds]].label.values
                    )
            preds = admc.predict_proba( df_with_batch.loc[df.index[test_inds]].values[:, 1:] )

            all_base_rocs.append(roc_auc_score(
                                            md.loc[md.index[test_inds]].label.values, 
                                            preds[:, 1]
                                            ) )

            np.random.seed(seed)
            torch.manual_seed(seed)

            admcwcs=AdaptationDebiasMClassifierWithinControlSimilarity()
            admcwcs.fit(df_with_batch.loc[df.index[train_inds]].values,
                     md.loc[md.index[train_inds]].label.values
                    )
            preds = admcwcs.predict_proba( df_with_batch.loc[df.index[test_inds]].values[:, 1:] )


            all_conly_rocs.append(roc_auc_score(
                                            md.loc[md.index[test_inds]].label.values, 
                                            preds[:, 1]
                                            ) )


            print('Base:')
            print(all_base_rocs)
            print('Control only:')
            print(all_conly_rocs)
        except:
            pass

        
        
    df, md = data_loading.load_HIVRC('../data/HIVRC/')
    df_with_batch = pd.concat([pd.Series( pd.Categorical(md.Study).codes, 
                                             index=md.index), df], axis=1)    
    y=md.label

    subb=df_with_batch.loc[:, 0].copy()
    subb.loc[subb==15]=14
    subb.loc[df_with_batch.loc[:, 0]==14]=15
    df_with_batch.loc[:, 0]=subb



    logo = LeaveOneGroupOut()
    all_base_rocs_hiv=[]
    all_conly_rocs_hiv=[]
    seed=0
    for train_inds, test_inds in logo.split(df, md, md.Study):
        try:

            np.random.seed(seed)
            torch.manual_seed(seed)

            admc=AdaptationDebiasMClassifier()
            admc.fit(df_with_batch.loc[df.index[train_inds]].values,
                     md.loc[md.index[train_inds]].label.values
                    )
            preds = admc.predict_proba( df_with_batch.loc[df.index[test_inds]].values[:, 1:] )

            all_base_rocs_hiv.append(roc_auc_score(
                                            md.loc[md.index[test_inds]].label.values, 
                                            preds[:, 1]
                                            ) )

            np.random.seed(seed)
            torch.manual_seed(seed)

            admcwcs=AdaptationDebiasMClassifierWithinControlSimilarity()
            admcwcs.fit(df_with_batch.loc[df.index[train_inds]].values,
                     md.loc[md.index[train_inds]].label.values
                    )
            preds = admcwcs.predict_proba( df_with_batch.loc[df.index[test_inds]].values[:, 1:] )


            all_conly_rocs_hiv.append(roc_auc_score(
                                            md.loc[md.index[test_inds]].label.values, 
                                            preds[:, 1]
                                            ) )


            print('Base:')
            print(all_base_rocs_hiv)
            print('Control only:')
            print(all_conly_rocs_hiv)
        except:
            pass

    summary_df = pd.DataFrame({'auROC':all_base_rocs_hiv+all_conly_rocs_hiv+\
                              all_base_rocs+all_conly_rocs,
                      'Method':['DEBIAS-M']*len(all_base_rocs_hiv) + \
                                ['DEBIAS-M control-weighting']*len(all_conly_rocs_hiv) + \
                                  ['DEBIAS-M']*len(all_base_rocs) + \
                                ['DEBIAS-M control-weighting']*len(all_conly_rocs),
                      'Task': ['HIVRC']*len(all_base_rocs_hiv) + \
                                ['HIVRC']*len(all_conly_rocs_hiv) + \
                                  ['CRC']*len(all_base_rocs) + \
                                ['CRC']*len(all_conly_rocs)
                          })

    summary_df.to_csv(os.path.join(results_path, 
                                   'wc-benchmark-results.csv') )
    
    return(None)


def save_plots(results_path='../results/within-control-comparison/',
               plots_path='../plots/within-control-comparison/'
              ):
    
    res = pd.read_csv(os.path.join(results_path, 
                               'wc-benchmark-results.csv'), 
                  index_col=0)
    
    plt.figure(figsize=(8,8))
    ax=sns.boxplot(x='Task', 
                   y='auROC',
                   hue='Method',
                   data=res,
                   fliersize=0, 
                   palette={'DEBIAS-M':global_palette['debias-m'], 
                            'DEBIAS-M control-weighting':\
                                     global_palette['debias-m'][:-2] + 'D4'   
                            }
                   )


    sns.swarmplot(x='Task', 
                  y='auROC',
                  hue='Method',
                  data=res,
                  dodge=True, 
                  color='black',
                  s=5, 
                  ax=ax
                  )

    pval_wilcox=wilcoxon(res.loc[res.Method=='DEBIAS-M'].auROC, 
                         res.loc[res.Method=='DEBIAS-M control-weighting'].auROC,
                         alternative='greater'
                         ).pvalue
    plt.title('Wicoxon p: {:.3e}'.format(pval_wilcox))
    plt.legend().remove()

    plt.savefig(
                os.path.join(plots_path, 
                             'wc-benchmark-results.pdf'
                             ), 
                format='pdf', 
                dpi=900, 
                bbox_inches='tight'
                )
    
    return(None)


def main():
    run_wc_sim_simulations(results_path='../results/within-control-comparison/')
    save_plots(results_path='../results/within-control-comparison/',
               plots_path='../plots/within-control-comparison/')
    
    
    
if __name__=='__main__':
    main()


    
    