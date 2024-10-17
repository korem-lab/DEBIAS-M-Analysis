
import numpy as np
import pandas as pd
import torch
import sys
sys.path.append('../v1-DEBIAS-M-Analysis/')
from General_functions import data_loading
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from debiasm_onebatch import DebiasMunibClassifier, rescale

def flatten(xss):
    return [x for xs in xss for x in xs]

def run_cv_benchmark(df, 
                     md, 
                     seed=42, 
                     n_splits=5, 
                     min_cv_counts=50
                     ):
    cv_study=[]
    baseline_aurocs=[]
    dmcu_aurocs=[]
    dmcu_var_aurocs=[]
    for sd in md.Study.unique():

        md_tmp=md.loc[md.Study==sd]
        df_tmp=rescale(df.loc[md_tmp.index])
        df_tmp=df_tmp.loc[:, (df_tmp>0).mean(axis=0)>0.05]
        if md_tmp.shape[0]>0:
            if (md_tmp.label.nunique()>1) and (md_tmp.shape[0] >= min_cv_counts ):
                np.random.seed(seed)
                lr_vals = cross_val_score(LogisticRegression(penalty='none', 
                                                             solver='newton-cg'
                                                             ), 
                                          X=df_tmp.values,
                                          y=md_tmp.label.values, 
                                          scoring='roc_auc', 
                                          cv=StratifiedKFold(n_splits=n_splits)
                                          )

                baseline_aurocs.append(lr_vals)

                dmuc=DebiasMunibClassifier()

                torch.manual_seed(seed)
                np.random.seed(seed)
                dmc_vals = cross_val_score(dmuc, 
                                           X=df_tmp.values, 
                                           y=md_tmp.label.values, 
                                           scoring='roc_auc', 
                                           cv=StratifiedKFold(n_splits=n_splits)
                                           )

                dmcu_aurocs.append(dmc_vals)
                cv_study.append([sd]*n_splits)


    results_df = pd.DataFrame({'Study':flatten(cv_study) + flatten(cv_study), 
                               'Group':['Raw']*len(flatten(baseline_aurocs)) + \
                                       ['DEBIAS-M-onebatch']*len(flatten(dmcu_aurocs)),
                               'auROC':flatten(baseline_aurocs)+flatten(dmcu_aurocs)
                               })
    
    return(results_df)
    
    
def main(results_csv_path='../results/within-batch-cv/linear-aurocs.csv'):
    all_res=[]
    for task in ['HIVRC', 'CRC']:
        df, md = data_loading.name_function_map[task]('../data/{}'.format(task))
        results_df = run_cv_benchmark(df, md) 
        results_df['Task']=task
        all_res.append(results_df)
    pd.concat(all_res).to_csv(results_csv_path)
    
    
    all_res = pd.read_csv('../results/within-batch-cv/linear-aurocs.csv', index_col=0)

    plt.figure(figsize=(8,8))
    ax=sns.boxplot(x='Task', 
                   y='auROC', 
                   hue='Group', 
                   data=all_res.sort_values('auROC'), 
                   palette={'Raw':global_palette['linear'], 
                               'DEBIAS-M-onebatch':global_palette['debias-m']}, 
                   hue_order=['Raw', 'DEBIAS-M-onebatch'], 
                   fliersize=0
                   )

    sns.swarmplot(x='Task', 
                  y='auROC', 
                  hue='Group', 
                  data=all_res.sort_values('auROC'), 
                  color='black', 
                  hue_order=['Raw', 'DEBIAS-M-onebatch'], 
                  s=5, 
                  dodge=True
                  )
    ax.legend().remove()
    pval=wilcoxon(
                all_res.loc[(all_res.Group=='DEBIAS-M-onebatch') ].auROC,
                all_res.loc[(all_res.Group=='Raw') ].auROC, 
                alternative='greater'
                ).pvalue
    plt.title('Within-study CV\n'+\
              'Wilcoxon p: {:.3f}'.format(pval))


    plt.savefig('../plots/within-batch-cv/5-fold-performance.pdf', 
                format='pdf', 
                dpi=900, 
                bbox_inches='tight'
                )
    
    
if __name__=='__main__':
    main()
    