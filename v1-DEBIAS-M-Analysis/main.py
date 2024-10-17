
import pandas as pd
import os
from General_functions.data_loading import name_function_map
name_function_map['CRC_with_labels'] = name_function_map['CRC']
from General_functions import all_classification_pipelines #import end_to_end_run
from General_functions import plotting #import save_roc_boxplots

## the below functions include analyses specific to a particular dataset
from CRC.main import CRC_functions
from HIVRC.main import HIVRC_functions
from Cervix import run_cervix_analysis
from Metabolites.main import Metabolites_functions
CRC_with_labels_functions = CRC_functions ## both of these are just placeholders

def flatten(l):
    return [item for sublist in l for item in sublist]


def batch_weight_feature_and_nbatchpairs_scaling(strength, df_with_batch):
    nbs = df_with_batch.iloc[:, 0].nunique()
    w = nbs * (nbs - 1) / 2
    return(strength /( w * ( df_with_batch.shape[1] - 1 ) ) )



def run_analyses(task, 
                 data_path_map, 
                 result_directories, 
                 rerun_other_batch_corrections=False):
    
    if task=='Cervix':
        run_cervix_analysis.main()
        return(None)
    
    
    # load data
    dataset=name_function_map[task](data_path_map[task])
    
    df = dataset[0]
    md = dataset[1]
    
#     df.to_csv('tmp/data.csv')
#     md.to_csv('tmp/metadata.csv')
    
    conq_out=pd.read_csv('../data/{}/conqur-out.csv'.format(task), index_col=0).round(5)
    comb_out=pd.read_csv('../data/{}/combat-out.csv'.format(task), index_col=0)

    if os.path.exists('../data/{}/voom-snm-out.csv'.format(task)):
        snm_out = pd.read_csv( '../data/{}/voom-snm-out.csv'.format(task), index_col=0 ).astype(float)

    else:
        snm_out = None
        
        
    if os.path.exists('../data/{}/MMUPHin_out.csv'.format(task)):
        mup_out = pd.read_csv( '../data/{}/MMUPHin_out.csv'.format(task), index_col=0 ).astype(float)

    else:
        mup_out = None
        
    if os.path.exists('../data/{}/PLSDAbatch_out.csv'.format(task)):
        pls_out = pd.read_csv( '../data/{}/PLSDAbatch_out.csv'.format(task), index_col=0 ).astype(float)

    else:
        snpls_outm_out = None
        
        
    if os.path.exists('../data/{}/percnorm_out.csv'.format(task)):
        perc_out = pd.read_csv( '../data/{}/percnorm_out.csv'.format(task), index_col=0 ).astype(float)

    else:
        snm_out = None

    comb_out.index=conq_out.index

    if snm_out is not None:
        snm_out.index = conq_out.index
        snm_out = snm_out.loc[md.index]

    conq_out=conq_out.loc[md.index]
    comb_out=comb_out.loc[md.index]
    mup_out=mup_out.loc[md.index]
    pls_out=pls_out.loc[md.index]
    perc_out=perc_out.loc[md.index].fillna(1e-6)
        
    df_with_batch = pd.concat([pd.Series( pd.Categorical(md.Study).codes, 
                                         index=md.index), df], axis=1)    
    y=md.label
    inds=df.sum(axis=1) > 10
    comb_out.index = df_with_batch.index
    
    if task=='Metabolites':
        eval('{}_functions'.format(task))(md, df_with_batch, conq_out, comb_out)
        return(None)

    
#     results_dict = all_classification_pipelines.run_predictions(y.loc[inds], 
#                                                                 df_with_batch.loc[inds], 
#                                                                 df_conqur = conq_out.loc[inds], 
#                                                                 df_combat = comb_out.loc[inds],
#                                                                 do_clr_transform=False,
#                                              b_str= batch_weight_feature_and_nbatchpairs_scaling( 1e4, df_with_batch ),
#                                                                 df_snm = [ snm_out.loc[inds] if snm_out is not None
#                                                                           else None][0], 
#                                                                 df_mup= [ mup_out.loc[inds] 
#                                                                  if mup_out is not None
#                                                                           else None][0],
#                                                                  df_perc= [ perc_out.loc[inds] 
#                                                                   if perc_out is not None
#                                                                            else None][0],
#                                                                 df_pls= [ pls_out.loc[inds] 
#                                                                  if pls_out is not None
#                                                                           else None][0],
#                                                                 min_epochs=25
#                                                                 )
    
#     studies = pd.concat([md.Study, df_with_batch[0]], axis=1).drop_duplicates().sort_values(0)
    
#     print(results_dict)
    
#     summary_auroc_df = pd.DataFrame({'auROC': flatten( [b for a,b in \
#                                                        results_dict['aurocs'].items()] ), 
#                                      'Group': flatten( [ [a]*len( results_dict['aurocs']['linear'])
#                                                       for a in results_dict['aurocs']]),
#                                      'auPR': flatten( [b for a,b in \
#                                                        results_dict['auprs'].items()] ),
#                                     })

#     studies = pd.concat([md.Study, df_with_batch[0]], axis=1).drop_duplicates()\
#                         .reset_index()
#     summary_auroc_df = pd.concat([summary_auroc_df, 
#                                       studies], 
#                                       axis=1
#                                     )

#     # save the boxplot
#     plotting.produce_auroc_boxplot(summary_auroc_df,
#                 out_path='../results/{}/auroc-boxplot-linear.pdf'.format(task), 
#                                   sig_dict=results_dict['is_sig']
#                                   )
    
    
#     plotting.produce_auroc_boxplot(summary_auroc_df,
#                 out_path='../results/{}/auroc-boxplot-linear.pdf'.format(task), 
#                                   hide_axes=False, 
#                                   sig_dict=results_dict['is_sig']
#                                   )
    
#     ## load plsda(clr) 
    if os.path.exists('../data/{}/CLR-PLSDAbatch_out.csv'.format(task)):
        plsclr = pd.read_csv( '../data/{}/CLR-PLSDAbatch_out.csv'.format(task), index_col=0 ).astype(float)
    
    results_dict = all_classification_pipelines.run_predictions(y.loc[inds], 
                                                                df_with_batch.loc[inds], 
                                                                conq_out.loc[inds], 
                                                                comb_out.loc[inds],
                                                                do_clr_transform=True,
                                                                df_mup= [ mup_out.loc[inds] 
                                                                 if mup_out is not None
                                                                          else None][0],
                                                                df_perc= [ perc_out.loc[inds] 
                                                                 if perc_out is not None
                                                                           else None][0],
                                                                df_pls=plsclr,
                                                                min_epochs=25,
                      b_str= batch_weight_feature_and_nbatchpairs_scaling( 1e3, df_with_batch )
                                                                )
    
    # studies = pd.concat([md.Study, df_with_batch[0]], axis=1).drop_duplicates().sort_values(0)

    
    
    summary_auroc_df = pd.DataFrame({'auROC': flatten( [b for a,b in \
                                                       results_dict['aurocs'].items()] ), 
                                     'Group': flatten( [ [a]*len( results_dict['aurocs']['linear'])
                                                      for a in results_dict['aurocs']]), 
                                     'auPR': flatten( [b for a,b in \
                                                       results_dict['auprs'].items()] ),
                                    })

    studies = pd.concat([md.Study, df_with_batch[0]], axis=1).drop_duplicates()\
                         .reset_index()
    summary_auroc_df = pd.concat([summary_auroc_df, 
                                      studies], 
                                      axis=1
                                    )

    # save the boxplot
    plotting.produce_auroc_boxplot(summary_auroc_df,
                out_path='../results/{}/auroc-clr-boxplot-linear.pdf'.format(task), 
                                  sig_dict=results_dict['is_sig'])
    
    # save the boxplot
    plotting.produce_auroc_boxplot(summary_auroc_df,
                out_path='../results/{}/auroc-clr-boxplot-linear.pdf'.format(task), 
                                  hide_axes=False, 
                                  sig_dict=results_dict['is_sig'])
    
#     eval('{}_functions'.format(task))(md, df_with_batch, conq_out, comb_out)
    
    return(None)
            


def main(task_list = [
#                        'Metabolites',
                        'Cervix',
                        'CRC',
                        'HIVRC', 
#                        'CRC_with_labels', 
#                        'HIVRC', 
                     ] ):
    ## for data setup
    data_path_map = {a:'../data/{}'.format(a) for a in task_list }
    
    ## for saving locations
    result_directories = {a:'../results/{}'.format(a) for a in task_list }
    
    for task in task_list:
        ## run analyses and store results
        run_analyses(task,
                     data_path_map,
                     result_directories)
    print('Successfully completed all analyses!')
    return(None)


if __name__=='__main__':
    main()
    
    
    