#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from Cervix.run_cervix_rf import execute_cervix_rf_run

from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from General_functions import plotting

import matplotlib.pyplot as plt
import seaborn as sns

from Cervix import run_cervix_rf
from debiasm import DebiasMClassifier
from General_functions.all_classification_pipelines import rescale
import torch
import torch.nn.functional as F
from main import batch_weight_feature_and_nbatchpairs_scaling
from skbio.diversity.alpha import shannon, observed_otus
from scipy.stats import wilcoxon

def main():


    carcinoma_rocs = execute_cervix_rf_run('Cervix-carcinoma', seed=1)
    cin_rocs = execute_cervix_rf_run('Cervix-CIN')


    pd.concat( [ pd.DataFrame.from_dict(a) for a in carcinoma_rocs ] )\
                    .to_csv('../results/Cervix-carcinoma/carcinoma-random-forest-predictions.csv')


    pd.concat( [ pd.DataFrame.from_dict(a) for a in cin_rocs ] )\
                    .to_csv('../results/Cervix-CIN/CIN-random-forest-predictions.csv')


    ca_df = pd.read_csv('../results/Cervix-carcinoma/carcinoma-random-forest-predictions.csv')
    ca_df = ca_df.loc[ca_df.test_batch_n < ca_df.test_batch_n.max()]


    cin_df = pd.read_csv('../results/Cervix-CIN/CIN-random-forest-predictions.csv')
    linear_benchmarks = pd.read_csv('../results/Cervix-CIN/auroc-boxplot-linear.csv')

    carcinoma_linear = pd.read_csv('../results/Cervix-carcinoma/linear-benchmark-rocs.csv', 
                              index_col=0)

    carcinoma_linear = carcinoma_linear.loc[carcinoma_linear.Group.str.contains('Raw|DEBIAS')]
    carcinoma_linear.Group = carcinoma_linear.Group.str.replace('Raw', 'Linear')


    plt.subplots(1,
                 figsize=(11.02, 10),
                 dpi=500)

    tbl = ca_df.copy()
    roc_dfs = []

    for nm in tbl.model.unique():
        fpr,tpr,_ = roc_curve(tbl.loc[tbl.model==nm]['True'],
                              tbl.loc[tbl.model==nm]['Pred'],
                              drop_intermediate=False
                             )

        name = nm.replace(' Tuning', '').replace(' Tuned', ' model')\
                        .replace('Raw', 'Linear (no correction)')\
                        .replace(' model', '')  + ' (auROC = {:.2f})'.format(auc(fpr, tpr))



        roc_dfs.append( pd.DataFrame({'TPR':tpr, 
                                          'FPR':fpr, 
                                          'Group':[name]*tpr.shape[0]
                                         }) )

    combined=pd.concat(roc_dfs)

    combined = pd.concat([combined, 
                          carcinoma_linear
                         ])

    ax=sns.lineplot(x='FPR', 
                    y='TPR', 
                    data=combined.reset_index(drop=True), 
                    hue='Group',
                    palette = {'Random Forest (auROC = 0.63)':plotting.global_palette['linear'], 
                               'DEBIAS-M into RF (auROC = 0.84)':plotting.global_palette['debias-m'],
                               'Linear (no correction) (auROC = 0.68)':'brown',
                               'DEBIAS-M (auROC = 0.85)':'lightblue'
                              },
                    hue_order=['Linear (no correction) (auROC = 0.68)', 
                           'DEBIAS-M (auROC = 0.85)', 
                           'Random Forest (auROC = 0.63)', 
                           'DEBIAS-M into RF (auROC = 0.84)'],
                   linewidth=5, 
                    ci=0
               )

    plt.ylim(0,1.005)
    plt.xlim(0,1)
    sns.lineplot([0,1], [0,1], color='black', linewidth=5, ax=ax)
    hide_axes=True
    plt.legend(loc='lower right')
    if hide_axes:
    #     ax.get_legend().remove()
        ax.set(yticklabels=[], xticklabels=[], yticks=[0,1], xticks=[0,1])
        plt.xlabel(None)
        plt.ylabel(None)
        plt.title(None)

    plt.savefig('../results/Cervix-carcinoma/rf-carcinoma-roc.pdf', 
                bbox_inches='tight', 
                dpi=900, 
                format='pdf', 
               )

    dmpr = pd.read_csv('../results/Cervix-carcinoma/carcinoma-debiasm-linear-predictions.csv')
    lrpr = pd.read_csv('../results/Cervix-carcinoma/carcinoma-raw-linear-predictions.csv')

    dmpr['True'] = dmpr['label']
    lrpr['True'] = lrpr['label']

    plt.subplots(1,
                 figsize=(11.02, 10),
                 dpi=500)

    tbl = ca_df.copy()
    tbl = pd.concat([tbl, dmpr, lrpr])
    pr_dfs = []

    for nm in tbl.model.unique():
        tpr,fpr,_ = precision_recall_curve(tbl.loc[tbl.model==nm]['True'],
                              tbl.loc[tbl.model==nm]['Pred'],
                                          )

        name = nm.replace(' Tuning', '').replace(' Tuned', ' model')\
                        .replace('Raw', 'Linear (no correction)')\
                        .replace(' model', '')  + ' (auPR = {:.2f})'.format(auc(fpr, tpr))

        pr_dfs.append( pd.DataFrame({'TPR':tpr, 
                                          'FPR':fpr, 
                                          'Group':[name]*tpr.shape[0]
                                         }) )

    combined=pd.concat(pr_dfs)
    combined.FPR -= combined.TPR * 1e-4 ## forcing the plot to look stepwise like for standard PR plot
    ax=sns.lineplot(x='FPR', 
                    y='TPR', 
                    data=combined.reset_index(drop=True), 
                    hue='Group',
                    linewidth=5,
                    palette = {'Random Forest (auPR = 0.45)':plotting.global_palette['linear'], 
                               'DEBIAS-M into RF (auPR = 0.68)':plotting.global_palette['debias-m'],
                               'DEBIAS-M (auPR = 0.71)':'lightblue',
                               'Linear (no correction) (auPR = 0.63)':'brown',
                           'Class balance ({:.2f})'.format(tbl.label.mean()):'black'
                              },
                    hue_order=['Linear (no correction) (auPR = 0.63)', 
                               'DEBIAS-M (auPR = 0.71)', 
                               'Random Forest (auPR = 0.45)', 
                               'DEBIAS-M into RF (auPR = 0.68)'
                              ],
                    ci=0, 
                    estimator=None
               )

    sns.lineplot([0, 1], 
               [tbl.label.mean(), 
                tbl.label.mean()], 
               linewidth = 5, 
               hue = ['Class balance ({:.2f})'.format(tbl.label.mean()) ]*2,
               palette={'Class balance ({:.2f})'.format(tbl.label.mean()):'black'},
               linestyle = '--', 
               ax=ax)

    plt.ylim(0,1.005)
    plt.xlim(0,1)

    leg = ax.legend()
    leg_lines = leg.get_lines()
    leg_lines[4].set_linestyle("--")

    hide_axes=True
    plt.legend(loc='lower right', fontsize=24)
    if hide_axes:
    #     ax.get_legend().remove()
        ax.set(yticklabels=[], xticklabels=[], yticks=[0,1], xticks=[0,1])
        plt.xlabel(None)
        plt.ylabel(None)
        plt.title(None)

    plt.savefig('../results/Cervix-carcinoma/rf-carcinoma-pr.pdf', 
                bbox_inches='tight', 
                dpi=900, 
                format='pdf', 
               )
    linear_benchmarks = pd.read_csv('../results/Cervix-CIN/auroc-boxplot-linear.csv')

    linear_benchmarks['model'] = linear_benchmarks['Group']
    linear_benchmarks['Test batch'] = ''
    linear_benchmarks['auroc'] = linear_benchmarks['auROC']
    linear_benchmarks = linear_benchmarks.loc[linear_benchmarks.model.isin(
                                    ['linear', 'debias-m'])]

    linear_benchmarks['Test batch'] = list( linear_benchmarks['Study'].dropna()) * 2

    plt.subplots(1,
                 figsize=(11.02, 10),
                 dpi=500)
    plot_data = pd.concat( [ pd.DataFrame({'Test batch':b['test_batch'],
                                           'model':b['model'],
                                           'auroc':roc_auc_score(b['True'], 
                                                                 b['Pred']) 
                                           }, 
                                                  index=[0])
                                       for b in cin_rocs ]).reset_index(drop=True) 


    plot_data = pd.concat([plot_data, 
                           linear_benchmarks[plot_data.columns] 
                          ])


    ax = sns.boxplot( x = 'model', 
                 y = 'auroc', 
                 palette = {'Random Forest Tuned':plotting.global_palette['linear'], 
                            'DEBIAS-M into RF Tuned':plotting.global_palette['debias-m'],
                            'linear':'brown',
                            'debias-m':'lightblue'
                           },
                 order=['linear', 
                        'debias-m', 
                        'Random Forest Tuned', 
                        'DEBIAS-M into RF Tuned'],
                 data = plot_data, 
                fliersize=0
                        ) 


    sns.swarmplot(x = 'model', 
                  y = 'auroc', 
                  data = plot_data, 
                  hue='Test batch',
                  order=['linear', 
                        'debias-m', 
                        'Random Forest Tuned', 
                        'DEBIAS-M into RF Tuned'],
    #               color='black', 
                  size=10,
                  ax=ax
                  ) 

    plt.xticks(rotation=90)
    plt.ylabel('auROC')
    plt.xlabel(None)
    ax.legend().remove()
    plt.savefig('../results/Cervix-CIN/rf-auroc-boxplot.pdf',
                bbox_inches='tight', 
                dpi=900, 
                format='pdf', 
               )
               

    plt.subplots(1,
                 figsize=(11.02, 10),
                 dpi=500)

    ax = sns.boxplot( x = 'model', 
                 y = 'auroc', 
                 data = plot_data, 
                     palette = {'Random Forest Tuned':plotting.global_palette['linear'], 
                            'DEBIAS-M into RF Tuned':plotting.global_palette['debias-m'],
                            'linear':'brown',
                            'debias-m':'lightblue'
                           },
                 order=['linear', 
                        'debias-m', 
                        'Random Forest Tuned', 
                        'DEBIAS-M into RF Tuned'],
                fliersize=0
                        ) 


    sns.swarmplot(x = 'model', 
                  y = 'auroc', 
                  data = plot_data, 
                  hue='Test batch',
    #               color='black', 
                  order=['linear', 
                        'debias-m', 
                        'Random Forest Tuned', 
                        'DEBIAS-M into RF Tuned'],
                  size=10,
                  ax=ax
                  ) 

    if hide_axes:
        ax.set_xticks([])
        ax.set(#yticks=np.linspace(.1,.9,5),
               yticklabels=[])
        plt.xlabel(None)
        plt.ylabel(None)
        plt.title(None)


    ax.legend().remove()


    plt.savefig('../results/Cervix-CIN/rf-auroc-boxplot-no-axes.pdf',
                bbox_inches='tight', 
                dpi=900, 
                format='pdf', 
                )


    # ## Diversity inferences



    seed=1
    task='Cervix-CIN'
    
    pal = plotting.global_palette
    pal['DEBIAS-M'] = pal['debias-m']
    pal['Raw'] = pal['linear']


    np.random.seed(seed)
    torch.manual_seed(seed)
    df_, md = run_cervix_rf.load_data(task)

    all_runs=[]

    df_with_batch = pd.concat([pd.Series( pd.Categorical( md['Study'] ).codes, 
                                      index=md.index ),
                                df_.loc[md.index]], axis=1)

    y=md[['label', 'Study']]

    inds = y['label'] != -1
    y=y.loc[inds]
    df_with_batch = df_with_batch.loc[inds]

    actual_str = batch_weight_feature_and_nbatchpairs_scaling(1e4, 
                                                              df_with_batch)

    batch = df_with_batch.iloc[:, 0].unique()[0]
    np.random.seed(seed)
    torch.manual_seed(seed)
    val_inds = df_with_batch.iloc[:, 0]==batch
    X_train, X_val = df_with_batch[~val_inds], df_with_batch[val_inds]
    y_train, y_val = y[~val_inds], y[val_inds]



    dmc = DebiasMClassifier(batch_str = actual_str, 
                            x_val=X_val.values
                            )
    dmc.fit(X_train.values, y_train.iloc[:, 0].values)

    x=torch.tensor(df_with_batch.values).float()
    batch_inds, x = x[:, 0], x[:, 1:]

    x = F.normalize( torch.pow(2, dmc.model.batch_weights[batch_inds.long()] ) * x, p=1 )
    df_wb_tmp=df_with_batch.copy()
    df_wb_tmp.iloc[:,1:]=x.detach().numpy()

    X_relabund = rescale( df_with_batch.iloc[:, 1:] )


    x_debias = pd.DataFrame( x.detach().numpy(), 
                             columns = X_relabund.columns, 
                             index = X_relabund.index 
                            )


    plt.subplots(1,
                 figsize=(5.51, 10),
                 dpi=500)
    ax = sns.boxplot(y = [shannon(a, base=2) for a in X_relabund.values] +                 [shannon(a, base=2) for a in x_debias.values], 
                x = ['Raw']*X_relabund.shape[0] + \
                        ['DEBIAS-M'] * x_debias.shape[0], 
                     fliersize=0, 
                     palette=pal

               )

    sns.swarmplot(y = [shannon(a, base=2) for a in X_relabund.values] +                 [shannon(a, base=2) for a in x_debias.values], 
                   x = ['Raw']*X_relabund.shape[0] + \
                          ['DEBIAS-M'] * x_debias.shape[0], 
                  color='black',
                  ax=ax
               )
    plt.title('DEBIAS-M increases shannon diversities\nWilcoxon $p<10^{-12}$')

    plt.savefig('../results/Cervix-CIN/shannon-diversities.pdf', 
                format='pdf', 
                dpi=900,
                bbox_inches='tight'
               )



    plt.subplots(1,
                 figsize=(4, 10),
                 dpi=500)
    ax = sns.boxplot(y = [shannon(a, base=2) for a in X_relabund.values] +                 [shannon(a, base=2) for a in x_debias.values], 
                x = ['Raw']*X_relabund.shape[0] + \
                        ['DEBIAS-M'] * x_debias.shape[0], 
                     fliersize=0, 
                     palette=pal

               )

    sns.swarmplot(y = [shannon(a, base=2) for a in X_relabund.values] +                 [shannon(a, base=2) for a in x_debias.values], 
                   x = ['Raw']*X_relabund.shape[0] + \
                          ['DEBIAS-M'] * x_debias.shape[0], 
                  color='black',
                  ax=ax
               )

    plt.ylabel(None)
    plt.xlabel(None)
    ax.set(yticklabels=[], 
           xticklabels=[]
           )

    plt.savefig('../results/Cervix-CIN/shannon-diversities-no-axes.pdf', 
                format='pdf', 
                dpi=900,
                bbox_inches='tight'
               )

    plt.show()



    plt.subplots(1,
                 figsize=(5.51, 10),
                 dpi=500)
    ax = sns.boxplot(y = [observed_otus(a) for a in X_relabund.values] +                 [observed_otus(a) for a in x_debias.values], 
                x = ['Raw']*X_relabund.shape[0] + \
                        ['DEBIAS-M'] * x_debias.shape[0], 
                     fliersize=0, 
                     palette=pal

               )

    sns.swarmplot(y = [observed_otus(a) for a in X_relabund.values] +                 [observed_otus(a) for a in x_debias.values], 
                   x = ['Raw']*X_relabund.shape[0] + \
                          ['DEBIAS-M'] * x_debias.shape[0], 
                  color='black',
                  ax=ax
               )


    plt.savefig('../results/Cervix-CIN/species-presence.pdf', 
                format='pdf', 
                dpi=900,
                bbox_inches='tight'
               )


    plt.subplots(1,
                 figsize=(4, 10),
                 dpi=500)
    ax = sns.boxplot(y = [observed_otus(a) for a in X_relabund.values] +                 [observed_otus(a) for a in x_debias.values], 
                x = ['Raw']*X_relabund.shape[0] + \
                        ['DEBIAS-M'] * x_debias.shape[0], 
                     fliersize=0, 
                     palette=pal

               )

    sns.swarmplot(y = [observed_otus(a) for a in X_relabund.values] +                 [observed_otus(a) for a in x_debias.values], 
                   x = ['Raw']*X_relabund.shape[0] + \
                          ['DEBIAS-M'] * x_debias.shape[0], 
                  color='black',
                  ax=ax
               )

    plt.ylabel(None)
    plt.xlabel(None)
    ax.set(yticklabels=[], 
           xticklabels=[]
           )

    plt.savefig('../results/Cervix-CIN/species-presence-no-axes.pdf', 
                format='pdf', 
                dpi=900,
                bbox_inches='tight'
               )

if __name__=='__main__':
    main()