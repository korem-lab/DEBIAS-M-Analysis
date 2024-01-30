
import numpy as np
import pandas as pd
from General_functions import all_classification_pipelines, plotting

def flatten(l):
    return [item for sublist in l for item in sublist]

def carcinoma_func(md):
    md['label']=md['carcinoma'].fillna(False).astype(bool)
    return(md)

def CIN_func(md):
    md['label']=md['CIN']!='normal'
    return(md)

task_mapping_dict = {
#                     'Cervix-carcinoma': carcinoma_func, ## this one is run in the separate file
                    'Cervix-CIN':CIN_func
                    }

def batch_weight_feature_and_nbatchpairs_scaling(strength, df_with_batch):
    nbs = df_with_batch.iloc[:, 0].nunique()
    w = nbs * (nbs - 1) / 2
    return(strength /( w * ( df_with_batch.shape[1] - 1 ) ) )

def main():

    
    for task in task_mapping_dict:
        
        df = pd.read_csv('../data/Cervix/data.csv', index_col=0)#.drop(['group', 'study'], axis=1)
        md = pd.read_csv('../data/Cervix/metadata.csv', index_col=0).loc[df.index]

        combat = pd.read_csv('../data/Cervix/combat-out.csv', index_col=0)
        conqur = pd.read_csv('../data/Cervix/conqur-out.csv', index_col=0)+1e-5 ## otherwise we get nans in some pipelines
        snm = pd.read_csv('../data/Cervix/voom-snm-out.csv', index_col=0)
        snm.index=combat.index
        
        md['Study'] = md.study
        md['Covariates'] = md.Age_Range


        md=task_mapping_dict[task](md)
        df=df.loc[md.index].round(5) ## adding this resolved discrepancies
                                ## between results on local machines

        df_with_batch = pd.concat([pd.Series( pd.Categorical(md.Study).codes, 
                                             index=md.index), df], axis=1)    
        y=md.label
        inds= ( df.sum(axis=1)*100 ) > 10

        conq_out = conqur.loc[df.index]
        comb_out = combat.loc[df.index]
        comb_out.index = df_with_batch.index
        snm_out = snm.loc[df.index]
        
        
        actual_str = batch_weight_feature_and_nbatchpairs_scaling(1e4, 
                                                                  df_with_batch.loc[inds])
   

        results_dict = all_classification_pipelines.run_predictions(y.loc[inds], 
                                                                    df_with_batch.loc[inds], 
                                                                    conq_out.loc[inds], 
                                                                    comb_out.loc[inds],
                                                                    do_clr_transform=False,
                                                                    learning_rate=.005,
                                                                    b_str=actual_str,
                                                                    min_epochs=25,
                                                                    df_snm = snm_out.loc[inds]
                                                                   )

        studies = pd.concat([md.Study, df_with_batch[0]], axis=1).drop_duplicates()\
                        .sort_values(0).reset_index()
        

        summary_auroc_df = pd.DataFrame({'auROC': flatten( [b for a,b in \
                                                           results_dict['aurocs'].items()] ), 
                                         'Group': flatten( [ [a]*len( results_dict['aurocs']['linear'])
                                                          for a in results_dict['aurocs']])
                                        })
        
        summary_auroc_df = pd.concat([summary_auroc_df, 
                                      studies], 
                                      axis=1
                                    )
        # save the boxplot
        plotting.produce_auroc_boxplot(summary_auroc_df,
                    out_path='../results/{}/auroc-boxplot-linear.pdf'.format(task), 
                                      sig_dict=results_dict['is_sig'])
        
        plotting.produce_auroc_boxplot(summary_auroc_df,
                    out_path='../results/{}/auroc-boxplot-linear.pdf'.format(task), 
                                       hide_axes=False,
                                      sig_dict=results_dict['is_sig'])
        
        
        results_dict = all_classification_pipelines.run_predictions(y.loc[inds], 
                                                                    df_with_batch.loc[inds], 
                                                                    conq_out.loc[inds], 
                                                                    comb_out.loc[inds],
                                                                    do_clr_transform=True,
                                                                    learning_rate=.005,
                                                                    b_str=actual_str/10,
                                                                    min_epochs=25,
                                                                   )

        studies = pd.concat([md.Study, df_with_batch[0]], axis=1).drop_duplicates().sort_values(0)

        summary_auroc_df = pd.DataFrame({'auROC': flatten( [b for a,b in \
                                                           results_dict['aurocs'].items()] ), 
                                         'Group': flatten( [ [a]*len( results_dict['aurocs']['linear'])
                                                          for a in results_dict['aurocs']]),
                                        })
        # save the boxplot
        plotting.produce_auroc_boxplot(summary_auroc_df,
                    out_path='../results/{}/auroc-clr-boxplot-linear.pdf'.format(task), 
                                      sig_dict=results_dict['is_sig'])
        
        plotting.produce_auroc_boxplot(summary_auroc_df,
                    out_path='../results/{}/auroc-clr-boxplot-linear.pdf'.format(task), 
                                       hide_axes=False,
                                      sig_dict=results_dict['is_sig'])
        
        
        
if __name__=='__main__':
    main()
       