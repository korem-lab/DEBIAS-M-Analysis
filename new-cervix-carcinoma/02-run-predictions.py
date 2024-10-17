

import numpy as np
import pandas as pd
from debiasm import DebiasMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import sys
sys.path.append('../v1-DEBIAS-M-Analysis/')
sys.path.append('../v1-DEBIAS-M-Analysis/General_functions')
from General_functions import data_loading, plotting
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from debiasm.torch_functions import rescale
import torch
import os

def batch_weight_feature_and_nbatchpairs_scaling(strength, df_with_batch):
    nbs = df_with_batch.iloc[:, 0].nunique()
    w = nbs * (nbs - 1) / 2
    return(strength /( w * ( df_with_batch.shape[1] - 1 ) ) )

def flatten(l):
    return [item for sublist in l for item in sublist]

from all_classification_pipelines import run_predictions

def main(out_path='cervix-all-new-data-with-all-methods.pdf'):
    
    
    md = pd.read_csv('processed-full-metadata.csv', index_col=0).reset_index(drop=True)
    df = pd.read_csv('processed-full-data.csv', index_col=0).reset_index(drop=True)
    
    df_with_batch = pd.concat([pd.Series( pd.Categorical(md.Study).codes, 
                                             index=md.index), df], axis=1) 
    df_conqur = pd.read_csv('tmp-datasets-for-R/conqur-out.csv', index_col=0)
    df_combat =  pd.read_csv('tmp-datasets-for-R/combat-out.csv', index_col=0)
    df_snm = pd.read_csv('tmp-datasets-for-R/voom-snm-out.csv', index_col=0)
    df_mup =  pd.read_csv('tmp-datasets-for-R/MMUPHin_out.csv', index_col=0)
    df_pls = pd.read_csv('tmp-datasets-for-R/PLSDAbatch_out.csv', index_col=0)
    df_perc =  pd.read_csv('tmp-datasets-for-R/percnorm_out.csv', index_col=0)
    df_combat.index = df_conqur.index
    df_snm.index = df_conqur.index
    
    
#     results_dict =  run_predictions(md.label, 
#                                     df_with_batch, 
#                                     df_conqur, 
#                                     df_combat, 
#                                     seed=123, 
#                                     do_clr_transform=False,
#                                     b_str=batch_weight_feature_and_nbatchpairs_scaling(1e4, df_with_batch),
#                                     df_snm=df_snm, 
#                                     df_mup=df_mup, 
#                                     df_pls=df_pls, 
#                                     df_perc=df_perc.fillna(1e-6)
#                                     )
    
#     print(results_dict)
#     studies = pd.concat([md.Study, df_with_batch[0]], axis=1).drop_duplicates().sort_values(0)

#     summary_auroc_df = pd.DataFrame({'auROC': flatten( [b for a,b in \
#                                                        results_dict['aurocs'].items()] ), 
#                                      'Group': flatten( [ [a]*len( results_dict['aurocs']['linear'])
#                                                       for a in results_dict['aurocs']]),
#                                      'auPR': flatten( [b for a,b in \
#                                                        results_dict['auprs'].items()] ),
#                                     })
    
    

#     studies = pd.concat([md.Study, df_with_batch.iloc[:, 0]], axis=1).drop_duplicates()\
#                         .reset_index()
    
#     studies = studies.loc[
#         studies.Study.isin(
#             [a for a in md.Study if md.loc[ md['Study']==a
#                                               ].label.max() > 0])
#                 ].reset_index(drop=True)
    
#     summary_auroc_df = pd.concat([summary_auroc_df, 
#                                       studies], 
#                                       axis=1
#                                     )
    
#     summary_auroc_df.to_csv('cervix-all-new-data-with-all-methods-results.csv')
    
#     # save the boxplot without axes
#     plotting.produce_auroc_boxplot(summary_auroc_df,
#                                    out_path=out_path, 
#                                    sig_dict=results_dict['is_sig'], 
#                                    )

#     # save the boxplot with axes
#     plotting.produce_auroc_boxplot(summary_auroc_df,
#                                    out_path=out_path, 
#                                    sig_dict=results_dict['is_sig'], 
#                                    hide_axes=False
#                                    )
    
    df_pls_clr = pd.read_csv('tmp-datasets-for-R/PLSDAbatch_out.csv', index_col=0)
    
    results_dict = run_predictions(md.label, 
                                   df_with_batch, 
                                   df_conqur, 
                                   df_combat,
                                   do_clr_transform=True,
                                   df_mup= [ df_mup
                                    if df_mup is not None
                                             else None][0],
                                   df_perc= [ df_perc.fillna(1e-6)
                                    if df_perc is not None
                                              else None][0],
                                   df_pls=df_pls_clr,
                                                                min_epochs=25,
                      b_str= batch_weight_feature_and_nbatchpairs_scaling( 1e3, df_with_batch )
                                                                )
    print(results_dict)
    
    
    summary_auroc_df = pd.DataFrame({'auROC': flatten( [b for a,b in \
                                                       results_dict['aurocs'].items()] ), 
                                     'Group': flatten( [ [a]*len( results_dict['aurocs']['linear'])
                                                      for a in results_dict['aurocs']]), 
                                     'auPR': flatten( [b for a,b in \
                                                       results_dict['auprs'].items()] ),
                                    })
    
    summary_auroc_df.to_csv('CLR-cervix-all-new-data-with-all-methods-results.csv')

    studies = pd.concat([md.Study, df_with_batch[0]], axis=1).drop_duplicates()\
                         .reset_index()
    summary_auroc_df = pd.concat([summary_auroc_df, 
                                      studies], 
                                      axis=1
                                    )

    # save the boxplot
    plotting.produce_auroc_boxplot(summary_auroc_df,
                                   out_path='CLR-'+out_path, 
                                   sig_dict=results_dict['is_sig'])
    
    # save the boxplot
    plotting.produce_auroc_boxplot(summary_auroc_df,
                                   out_path='CLR-'+out_path, 
                                   hide_axes=False, 
                                   sig_dict=results_dict['is_sig'])
                
    print('Successful run completed!')
    
    
    
if __name__=='__main__':
    main()
        
        