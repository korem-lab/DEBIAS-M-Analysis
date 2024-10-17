import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('../v1-DEBIAS-M-Analysis/')
from General_functions import data_loading, plotting
from sklearn.model_selection import LeaveOneGroupOut
from dm_sweighting import DebiasMSWClassifier
from debiasm import DebiasMClassifier
import torch
from sklearn.metrics import roc_auc_score
import os
from scipy.stats import wilcoxon
from General_functions.plotting import global_palette


def run_and_save_predictons(results_path='../results/study-weighting/'):
    df, md = data_loading.load_CRC('../data/CRC/')
    df_with_batch = pd.concat([pd.Series( pd.Categorical(md.Study).codes, 
                                             index=md.index), df], axis=1)    
    y=md.label

    logo = LeaveOneGroupOut()
    all_base_rocs=[]
    all_sw_rocs=[]
    seed=123
    for train_inds, test_inds in logo.split(df, md, md.Study):
        np.random.seed(seed)
        torch.manual_seed(seed)

        dmc=DebiasMClassifier(x_val=df_with_batch.loc[df.index[test_inds]].values)
        dmc.fit(df_with_batch.loc[df.index[train_inds]].values,
                 md.loc[md.index[train_inds]].label.values
                )
        preds = dmc.predict_proba( df_with_batch.loc[df.index[test_inds]].values )

        all_base_rocs.append(roc_auc_score(
                                        md.loc[md.index[test_inds]].label.values, 
                                        preds[:, 1]
                                        ) )

        np.random.seed(seed)
        torch.manual_seed(seed)

        dmcsw=DebiasMSWClassifier(x_val=df_with_batch.loc[df.index[test_inds]].values)
        dmcsw.fit(df_with_batch.loc[df.index[train_inds]].values,
                 md.loc[md.index[train_inds]].label.values
                )
        preds = dmcsw.predict_proba( df_with_batch.loc[df.index[test_inds]].values )


        all_sw_rocs.append(roc_auc_score(
                                        md.loc[md.index[test_inds]].label.values, 
                                        preds[:, 1]
                                        ) )


        print('Base:')
        print(all_base_rocs)
        print('Studysizeweighting:')
        print(all_sw_rocs)


    df, md = data_loading.load_HIVRC('../data/HIVRC/')
    df_with_batch = pd.concat([pd.Series( pd.Categorical(md.Study).codes, 
                                             index=md.index), df], axis=1)    
    y=md.label


    logo = LeaveOneGroupOut()
    all_base_rocs_hiv=[]
    all_sw_rocs_hiv=[]
    seed=123

    for train_inds, test_inds in logo.split(df, md, md.Study):
        try:
            np.random.seed(seed)
            torch.manual_seed(seed)

            dmc=DebiasMClassifier(x_val=df_with_batch.loc[df.index[test_inds]].values)
            dmc.fit(df_with_batch.loc[df.index[train_inds]].values,
                     md.loc[md.index[train_inds]].label.values
                    )
            preds = dmc.predict_proba( df_with_batch.loc[df.index[test_inds]].values )

            all_base_rocs_hiv.append(roc_auc_score(
                                            md.loc[md.index[test_inds]].label.values, 
                                            preds[:, 1]
                                            ) )

            np.random.seed(seed)
            torch.manual_seed(seed)

            dmcsw=DebiasMSWClassifier(x_val=df_with_batch.loc[df.index[test_inds]].values)
            dmcsw.fit(df_with_batch.loc[df.index[train_inds]].values,
                     md.loc[md.index[train_inds]].label.values
                    )
            preds = dmcsw.predict_proba( df_with_batch.loc[df.index[test_inds]].values )


            all_sw_rocs_hiv.append(roc_auc_score(
                                            md.loc[md.index[test_inds]].label.values, 
                                            preds[:, 1]
                                            ) )


            print('Base:')
            print(all_base_rocs_hiv)
            print('Studysizeweighting:')
            print(all_sw_rocs_hiv)
        except:
            pass



    summary_df = pd.DataFrame({'auROC':all_base_rocs_hiv+all_sw_rocs_hiv+\
                                  all_base_rocs+all_sw_rocs,
                          'Method':['DEBIAS-M']*len(all_base_rocs_hiv) + \
                                    ['DEBIAS-M per-sample-weighting']*len(all_sw_rocs_hiv) + \
                                      ['DEBIAS-M']*len(all_base_rocs) + \
                                    ['DEBIAS-M per-sample-weighting']*len(all_sw_rocs),
                          'Task': ['HIVRC']*len(all_base_rocs_hiv) + \
                                    ['HIVRC']*len(all_sw_rocs_hiv) + \
                                      ['CRC']*len(all_base_rocs) + \
                                    ['CRC']*len(all_sw_rocs)
                          })

    summary_df.to_csv(os.path.join(results_path, 
                                   'sw-benchmark-results.csv') )
    
    
def generate_plot(results_path='../results/study-weighting/',
                  plot_path='../plots/study-weighting/'
                  ):
    
    res = pd.read_csv(os.path.join(results_path, 
                               'sw-benchmark-results.csv'), 
                  index_col=0)
    
    plt.figure(figsize=(8,8))
    ax=sns.boxplot(x='Task', 
                   y='auROC',
                   hue='Method',
                   data=res,
                   fliersize=0, 
                   palette={'DEBIAS-M':global_palette['debias-m'], 
                            'DEBIAS-M per-sample-weighting':\
                                     global_palette['debias-m'][:-2] + 'A9'   
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
                         res.loc[res.Method=='DEBIAS-M per-sample-weighting'].auROC,
                         alternative='greater'
                         ).pvalue
    plt.title('Wicoxon p: {:.3e}'.format(pval_wilcox))
    plt.legend().remove()

    plt.savefig(
                os.path.join(plot_path, 
                             'sw-benchmark-results.pdf'
                             ), 
                format='pdf', 
                dpi=900, 
                bbox_inches='tight'
                )
    return(None)


def main(results_path='../results/study-weighting/',
         plot_path='../plots/study-weighting/'):
    
    run_and_save_predictons(results_path=results_path)
    generate_plot(results_path=results_path, 
                  plot_path=plot_path)
    
if __name__=='__main__':
    main()
                  

