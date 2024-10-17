

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
# from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
# from dm_regression import DebiasMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from debiasm.torch_functions import rescale
from sklearn.linear_model import LinearRegression
# import torch
import sys
sys.path.append('../v1-DEBIAS-M-Analysis/')
from General_functions import plotting
from General_functions import data_loading
# from sklearn.model_selection import GridSearchCV
# import dm_regression
# import dmc_multitask_regression 
import os

## setups for melonpann run
# rescale( np.log10( md[ cols[:-2] ] + 1 ) ).to_csv('melon-data/metabs_relabund.csv')
# X = df
# inds = ( ( X > 0 ).mean(axis=0) > .05 )
# X_relb = rescale(X.loc[:,inds])
# X_relb.to_csv('melon-data/microbs_relabund.csv')
# md.iloc[:, -6:].to_csv('melon-data/metadata.csv')




def get_melon_summary(md, cols):
    metab_trues = rescale( np.log10( md[ cols[:-2] ] + 1 ) )
    metab_trues.columns = pd.read_csv('../regression/melon-data/metab-R-column-names.csv', index_col=0).columns
    melon_preds=pd.read_csv('../regression/melon-data/melonpann_test_out/MelonnPan_Predicted_Metabolites.txt', 
                            index_col=0, sep='\t')
    metab_trues=metab_trues.loc[melon_preds.index]
    melon_perfs = pd.Series( [spearmanr(melon_preds[a], metab_trues[a]).correlation for a in melon_preds.columns] 
                     ).fillna(0)
    melon_preds=pd.read_csv('../regression/melon-data/melonpann_test_out/MelonnPan_Predicted_Metabolites.txt', 
                            index_col=0, sep='\t')
    metab_trues=metab_trues.loc[melon_preds.index]
    melon_perfs = pd.Series( [spearmanr(melon_preds[a], metab_trues[a]).correlation for a in melon_preds.columns] 
                     ).fillna(0)


    metab_trues = rescale( np.log10( md[ cols[:-2] ] + 1 ) )
    rform_cols=pd.read_csv('../regression/melon-data/metab-R-column-names.csv', index_col=0).columns

    return( pd.DataFrame({'Spearman R': melon_perfs.values, 
                          'Group':'MelonPann', 
                          'Metabolite':metab_trues.columns[ np.where( rform_cols.isin(melon_preds.columns) ) ]
                          })
            )


def get_predictive_perfs(df,
                         md,
                         cols, 
                         possible_feat_inds,
                         melon_res,
                         seed=42, 
                         min_presence_thresh=0):

    base_spears=[]
    dm_spears=[]
    selected_cols=[]
    
    for i in possible_feat_inds:
#     for i in range(cols.shape[0]-2):
    # for i in np.where( metab_trues.columns.isin( melon_preds.columns ) )[0]:

        if (md[cols[i]]>0 ).sum() > min_presence_thresh:

            selected_cols.append(cols[i])

            np.random.seed(seed)
            torch.manual_seed(seed)

            ss = StandardScaler()
            y=ss.fit_transform( np.log10( md[ [cols[i]] ] + 1 ) )[:, 0]
            X = rescale(df.values)



            ii = ( md[ cols[i] ]>-1e6 ).values ## use all remaining vals

            X=X[ii]
            grps=md.Study.values[ii]
            y=y[ii]

            inds = ( ( X > 0 ).mean(axis=0) > .05 )

            X=rescale(X[:, inds])

            train_nm='HSRR_JR003_mm_corrected'
            train_inds = np.where( grps==train_nm )[0]
            test_inds = np.where( grps!=train_nm )[0]

            lr=LinearRegression()
            lr.fit(X[train_inds], y[train_inds])
            print(y[train_inds].shape)


            base_spears.append( spearmanr(y[test_inds],
                                 lr.predict(X[test_inds])
                                 ) )

            X_with_batch=pd.DataFrame( 
                        np.hstack(( (grps==grps[0] ).astype(int)[:,np.newaxis], 
                                     X )) )

            dmr = dm_regression.DebiasMRegressor( x_val = X_with_batch.loc[test_inds] )

            dmr.fit(pd.DataFrame( X_with_batch.loc[train_inds]), 
                    y[train_inds])

            dm_spears.append(
                spearmanr(dmr.predict( (X_with_batch).values[test_inds] ), 
                           y[test_inds]
                           ) )
            print([ round( a[0], 2) for a in base_spears])
            print([ round( a[0], 2) for a in dm_spears])
    

    np.random.seed(seed)
    torch.manual_seed(seed)

    mdmr=dmc_multitask_regression.MultitaskDebiasMRegressor(
                                x_val= X_with_batch.loc[test_inds].values
                                )

    y_multi = pd.DataFrame(ss.fit_transform( np.log10( md[selected_cols] + 1 ) ), 
                           columns=selected_cols
                           )

    mdmr.fit(X_with_batch.loc[train_inds].values, 
             y_multi.loc[train_inds].values
             )

    multi_preds = mdmr.predict( X_with_batch.loc[test_inds].values )
    
    multi_spears = [ spearmanr(y_multi.values[test_inds][:, i], 
                 multi_preds[i][:,0]).correlation
                 for i in range(len(multi_preds)) ]
    
    
    python_res = pd.DataFrame({ 'Spearman R': [a[0] for a in base_spears] + 
                                          [a[0] for a in dm_spears] + \
                                                  multi_spears, 
                           'Group': ['Raw']*len(base_spears) + \
                                       ['DEBIAS-M']*len(dm_spears) + \
                                          ['Multitask\nDEBIAS-M']*len(multi_spears), 
                           'Metabolite':selected_cols*3
                          })
    
    return( pd.concat([melon_res, python_res]) )
    
    
def get_specific_method_predictive_perfs(df,
                                         md,
                                         cols, 
                                         method_nm,
                                         possible_feat_inds,
                                         melon_res,
                                         seed=42, 
                                         min_presence_thresh=0
                                         ):

    method_spears=[]
    selected_cols=[]
    
    for i in possible_feat_inds:

        if (md[cols[i]]>0 ).sum() > min_presence_thresh:

            selected_cols.append(cols[i])

            np.random.seed(seed)
            
            ss = StandardScaler()
            y=ss.fit_transform( np.log10( md[ [cols[i]] ] + 1 ) )[:, 0]
            X = df

            ii = ( md[ cols[i] ]>-1e6 ).values ## use all remaining vals

            X=X[ii]
            grps=md.Study.values[ii]
            y=y[ii]

            train_nm='HSRR_JR003_mm_corrected'
            train_inds = np.where( grps==train_nm )[0]
            test_inds = np.where( grps!=train_nm )[0]

            lr=LinearRegression()
            lr.fit(X[train_inds], y[train_inds])

            method_spears.append( spearmanr(y[test_inds],
                                 lr.predict(X[test_inds])
                                 ) )
    
    
    return( pd.DataFrame({'Spearman R': [a[0] for a in method_spears],
                               'Group': method_nm, 
                               'Metabolite':selected_cols
                              })
          )
            
        
        
def main(out_path_dir='../results/regression-all-methods/'):

    df, md = data_loading.load_Metabolites('../data/Metabolites/')

    cols = md.columns[3:]
    cols=cols[:-4]
    melon_res=get_melon_summary(md, cols)
    
    feat_ind_map = { 'melon':np.where( cols.isin( melon_res.Metabolite.values ) )[0],
                     'all':range(cols.shape[0]-2) }
    
    pths = ['tmp-datasets-for-R/percnorm_out.csv',
            'tmp-datasets-for-R/combat-out.csv',
            'tmp-datasets-for-R/conqur-out.csv',
            'tmp-datasets-for-R/MMUPHin_out.csv',
            'tmp-datasets-for-R/PLSDAbatch_out.csv',
            'tmp-datasets-for-R/voom-snm-out.csv'
            ]


    
    for min_thresh in [50, 0]:
        for possible_feat_inds in ['melon', 'all']:
    
            all_perfs=[]
            for a in pths:
                tt=pd.read_csv(a, index_col=0) 
                if a not in pths[-3:]:
                    tt=rescale(tt)
                else:
                    tt=tt

                ss=StandardScaler()
                tt=ss.fit_transform(tt)
                print(a)
                
                all_perfs.append( get_specific_method_predictive_perfs(tt,
                                             md,
                                             cols, 
                                             method_nm=a.split('/')[-1].split('-')[0].split('_')[0],
                                             possible_feat_inds=feat_ind_map[possible_feat_inds],
                                             melon_res=melon_res,
                                             min_presence_thresh=min_thresh
                                             )
                                )

            pd.concat(all_perfs).to_csv(os.path.join(out_path_dir, 
                                         'all-methods-vaginal-metab-min-thresh-{}-features-{}.csv'\
                                                 .format(min_thresh, possible_feat_inds)
                                        )
                           )
            
    
if __name__=='__main__':
    main()
    
    
    
    
    

