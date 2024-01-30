#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from debiasm.torch_functions import rescale
from sklearn.metrics import roc_curve
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

from General_functions.plotting import global_palette
from General_functions.delong import delong_roc_test

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
                    'Cervix-carcinoma': carcinoma_func,
                    }

def batch_weight_feature_and_nbatchpairs_scaling(strength, df_with_batch):
    nbs = df_with_batch.iloc[:, 0].nunique()
    w = nbs * (nbs - 1) / 2
    return(strength /( w * ( df_with_batch.shape[1] - 1 ) ) )

def main():

    for task in task_mapping_dict:


        df = pd.read_csv('../data/Cervix/data.csv', index_col=0)
        md = pd.read_csv('../data/Cervix/metadata.csv', index_col=0).loc[df.index]

        combat = pd.read_csv('../data/Cervix/combat-out.csv', index_col=0)
        conqur = pd.read_csv('../data/Cervix/conqur-out.csv', index_col=0) + 1e-5 ## we get a nan error otherwise
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

        studies = pd.concat([md.Study, df_with_batch[0]], axis=1).drop_duplicates().sort_values(0)

        summary_auroc_df = pd.DataFrame({'auROC': flatten( [b for a,b in                                                        results_dict['aurocs'].items()] ), 
                                         'Group': flatten( [ [a]*len( results_dict['aurocs']['linear'])
                                                          for a in results_dict['aurocs']])
                                        })



    preds_db = results_dict['models']['debias-m'][0]                .forward(torch.tensor(df_with_batch.loc[inds].values ).float() )[:, 1]                    .detach().numpy()


    roc_dfs = []
    bi = 0
    order_map={}
    for a in results_dict['models']:
        inds = md.loc[df_with_batch[0]==bi].index
        if a == 'debias-m':
            preds = results_dict['models'][a][0].forward(torch.tensor(df_with_batch.loc[inds].values ).float() )[:, 1]                    .detach().numpy()
            nm = 'DEBIAS-M'
            pd.concat([ md.loc[inds].label, 
                       pd.Series( preds, index=md.loc[inds].index, name='Pred' ), 
                       pd.Series( ['DEBIAS-M']*inds.shape[0],
                         index=md.loc[inds].index,
                         name='model' )
               ], axis=1
             ).to_csv('../results/Cervix-carcinoma/carcinoma-debiasm-linear-predictions.csv')
            
            
        elif a == 'linear':
            preds = results_dict['models'][a][0][1].predict_proba( 
                        results_dict['models'][a][0][0].transform(
                             rescale( df_with_batch.iloc[:, 1:].loc[df_with_batch[0]==bi]) ) )[:, 1]
            nm = 'Raw (no correction)'
            
            pd.concat([ md.loc[inds].label,
                             pd.Series( preds, index=md.loc[inds].index, name='Pred' ), 
                            pd.Series( ['Raw']*inds.shape[0],
                                     index=md.loc[inds].index,
                                      name='model' )
                   ], axis=1
                 ).to_csv('../results/Cervix-carcinoma/carcinoma-raw-linear-predictions.csv')

        elif a == 'conqur':
            preds = results_dict['models'][a][0][1].predict_proba( 
                results_dict['models'][a][0][0].transform( rescale( conq_out.loc[inds]) ) )[:, 1]
            nm='ConQuR'

        elif a == 'combat':
            preds = results_dict['models'][a][0][1].predict_proba( 
                            results_dict['models'][a][0][0].transform( rescale( comb_out.loc[inds]) ) )[:, 1]
            nm='ComBat'

        elif a == 'snm':
            preds = results_dict['models'][a][0][1].predict_proba( 
                results_dict['models'][a][0][0].transform( snm_out.loc[inds]) )[:, 1]
            nm='Voom-SNM'


        print(a)
        print(roc_auc_score(md.loc[inds].label, 
                             preds).round(2))

        fpr,tpr,_= roc_curve(md.loc[df_with_batch[0]==bi].label, 
                             preds, 
                             drop_intermediate=False)


        name = nm + ' (auROC = {:.2f})'.format(auc(fpr, tpr))
        global_palette[name] = global_palette[a]

        order_map[a] = name

        roc_dfs.append( pd.DataFrame({'TPR':tpr, 
                                      'FPR':fpr, 
                                      'Group':[name]*tpr.shape[0]
                                     })
                      )

    combined = pd.concat(roc_dfs)
    combined.to_csv('../results/Cervix-carcinoma/linear-benchmark-rocs.csv')
    
    order=['linear', 
           'combat', 
           'conqur', 
           'snm',
           'debias-m']
    plt.subplots(1,
                 figsize=(11.02, 10),
                 dpi=500)

    ax=sns.lineplot(x='FPR', 
                   y='TPR', 
                   data=combined.reset_index(drop=True), 
                   hue='Group',
                   hue_order = [order_map[a] for a in order],
                   palette=global_palette,
                   linewidth=5, 
                    ci=0
               )

    sns.lineplot([0,1], [0,1], color='black', linewidth=5, ax=ax)
    hide_axes=True
    plt.legend(loc='lower right')
    if hide_axes:
        ax.set(yticklabels=[], xticklabels=[], yticks=[0,1], xticks=[0,1])
        plt.xlabel(None)
        plt.ylabel(None)
        plt.title(None)

    plt.savefig('../results/Cervix-carcinoma/cervix-carcinoma-roc.pdf', 
                bbox_inches='tight', 
                dpi=900, 
                format='pdf', 
               )


    global_palette[ order_map['snm'] ] = 'pink'



    order=['linear', 
           'combat', 
           'conqur', 
           'snm',
           'debias-m']
    plt.subplots(1,
                 figsize=(11.02, 10),
                 dpi=500)

    ax=sns.lineplot(x='FPR', 
                   y='TPR', 
                   data=combined.reset_index(drop=True), 
                   hue='Group',
                   hue_order = [order_map[a] for a in order],#order,
                   palette=global_palette,
                   linewidth=5, 
                    ci=0
               )

    sns.lineplot([0,1], [0,1], color='black', linewidth=5, ax=ax)
    plt.ylim(0,1)
    plt.xlim(0,1)
    hide_axes=True
    plt.legend(loc='lower right')
    if hide_axes:
        ax.set(yticklabels=[], xticklabels=[], yticks=[0,1], xticks=[0,1])
        plt.xlabel(None)
        plt.ylabel(None)
        plt.title(None)


    plt.savefig('../results/Cervix-carcinoma/cervix-carcinoma-roc.pdf', 
                bbox_inches='tight', 
                dpi=900, 
                format='pdf', 
               )


    for task in task_mapping_dict:
        
        df = pd.read_csv('../data/Cervix/data.csv', index_col=0)#.drop(['group', 'study'], axis=1)
        md = pd.read_csv('../data/Cervix/metadata.csv', index_col=0).loc[df.index]

        combat = pd.read_csv('../data/Cervix/combat-out.csv', index_col=0)
        conqur = pd.read_csv('../data/Cervix/conqur-out.csv', index_col=0)+ 1e-5 ## we get a nan error otherwise
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


        actual_str = batch_weight_feature_and_nbatchpairs_scaling(1e3, 
                                                                  df_with_batch.loc[inds])


        results_dict = all_classification_pipelines.run_predictions(y.loc[inds], 
                                                                    df_with_batch.loc[inds], 
                                                                    conq_out.loc[inds], 
                                                                    comb_out.loc[inds],
                                                                    do_clr_transform=True,
                                                                    learning_rate=.005,
                                                                    b_str=actual_str,
                                                                    min_epochs=25,
    #                                                                 df_snm = snm_out.loc[inds]
                                                                   )

        studies = pd.concat([md.Study, df_with_batch[0]], axis=1).drop_duplicates().sort_values(0)

        summary_auroc_df = pd.DataFrame({'auROC': flatten( [b for a,b in                                                        results_dict['aurocs'].items()] ), 
                                         'Group': flatten( [ [a]*len( results_dict['aurocs']['linear'])
                                                          for a in results_dict['aurocs']])
                                        })
                                        
    preds_db = results_dict['models']['debias-m'][0]                .forward(torch.tensor(df_with_batch.loc[inds].values ).float() )[:, 1]                    .detach().numpy()

    df_with_batch=df_with_batch.loc[results_dict['clr_datasets']['raw'].index]
    md=md.loc[df_with_batch.index]

    roc_dfs = []
    bi = 0
    order_map={}
    for a in results_dict['models']:

        if a != 'snm':
            print(a)
            inds = md.loc[df_with_batch[0]==bi].index
            if a == 'debias-m':
                preds = results_dict['models'][a][0]                    .forward(torch.tensor(results_dict['clr_datasets']['raw'].loc[inds].values ).float() )[:, 1]                        .detach().numpy()
                nm = 'DEBIAS-M'
            elif a == 'linear':
                preds = results_dict['models'][a][0][1].predict_proba( 
                            results_dict['models'][a][0][0].transform(
                                 results_dict['clr_datasets']['raw'].iloc[:, 1:].loc[df_with_batch[0]==bi]) )[:, 1]
                nm = 'Raw (no correction)'

            elif a == 'conqur':
                preds = results_dict['models'][a][0][1].predict_proba( 
                    results_dict['models'][a][0][0].transform( results_dict['clr_datasets']['conqur'].loc[inds]) )[:, 1]
                nm='ConQuR'

            elif a == 'combat':
                preds = results_dict['models'][a][0][1].predict_proba( 
                                results_dict['models'][a][0][0].transform( results_dict['clr_datasets']['combat'].loc[inds]) )[:, 1]
                nm='ComBat'



            print(a)
            print(roc_auc_score(md.loc[inds].label, 
                                 preds).round(2))

            fpr,tpr,_= roc_curve(md.loc[df_with_batch[0]==bi].label, 
                                 preds, 
                                 drop_intermediate=False)


            name = nm + ' (auROC = {:.2f})'.format(auc(fpr, tpr))
            global_palette[name] = global_palette[a]

            order_map[a] = name

            roc_dfs.append( pd.DataFrame({'TPR':tpr, 
                                          'FPR':fpr, 
                                          'Group':[name]*tpr.shape[0]
                                         })
                          )



    combined = pd.concat(roc_dfs)

    order=['linear', 
           'combat', 
           'conqur', 
           'debias-m']
    plt.subplots(1,
                 figsize=(11.02, 10),
                 dpi=500)

    ax=sns.lineplot(x='FPR', 
                   y='TPR', 
                   data=combined.reset_index(drop=True), 
                   hue='Group',
                   hue_order = [order_map[a] for a in order],
                   palette=global_palette,
                   linewidth=5, 
                    ci=0
               )

    sns.lineplot([0,1], [0,1], color='black', linewidth=5, ax=ax)
    plt.ylim(0,1)
    plt.xlim(0,1)
    hide_axes=True
    plt.legend(loc='lower right')
    if hide_axes:
        ax.set(yticklabels=[], xticklabels=[], yticks=[0,1], xticks=[0,1])
        plt.xlabel(None)
        plt.ylabel(None)
        plt.title(None)


    plt.savefig('../results/Cervix-carcinoma/cervix-clr-carcinoma-roc.pdf', 
                bbox_inches='tight', 
                dpi=900, 
                format='pdf', 
               )

    
if __name__=='__main__':
    main()

