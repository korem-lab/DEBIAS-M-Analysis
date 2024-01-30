#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from General_functions import plotting
global_palette = plotting.global_palette
ticker=plotting.ticker
from scipy.stats import wilcoxon
def flatten(l):
    return [item for sublist in l for item in sublist]

def main():

    data=pd.read_csv('../results/Metabolites/Metab-auroc-boxplot-linear.csv', index_col=0)


    order = ['linear', 
             'combat', 
             'conqur', 
             'debias-m'
             ]

    plt.subplots(1,
                 figsize=(11.02, 10),
                 dpi=500)

    ax=sns.violinplot(x='Group', 
                      y='auROC', 
                      data=data, 
                      order=order,
                      hue_order = order,
                      palette=global_palette,
                      width=.75,
                      fliersize=0
                      )

    handles, labels = ax.get_legend_handles_labels()

    sns.stripplot(x='Group', 
                  y='auROC', 
                  data=data,
                  order=order,
                  hue_order = order,
                  dodge=True,
                  ax=ax, 
                  size=10, 
                  color='k',
                  alpha=.1
               )



    # plot vs scrub significance asterisks

    # 'worse/better than' test
    q=data[['Group']].drop_duplicates()
    q['sig_v_sbcmp_greater'] = q.apply(
                            lambda row: [ wilcoxon( 
                               data.loc[ (data.Group==row.Group) ]['auROC'].values, 
                               data.loc[ (data.Group=='debias-m' ) ]['auROC'].values, 
                    alternative='less').pvalue
                                    if row.Group!='debias-m' else 1][0],
                       axis=1)

    q['sig_v_sbcmp_less'] = q.apply(
                            lambda row: [ wilcoxon( 
                               data.loc[ (data.Group==row.Group) ]['auROC'].values, 
                               data.loc[ (data.Group=='debias-m' ) ]['auROC'].values, 
                    alternative='greater').pvalue
                                    if row.Group!='debias-m' else 1][0],
                       axis=1)



    q['lower_y_val'] = -0.025 # data.auROC.min() - 0.03
    q['lower_is_sig'] =  q.sig_v_sbcmp_greater < 0.05

    q['greater_y_val'] = -0.025 #data.auROC.max() + 0.03
    q['greater_is_sig'] =  q.sig_v_sbcmp_less < 0.05

    if sum(q.greater_is_sig > 0):
        sns.swarmplot(x='Group', 
                      y='greater_y_val', 
                      data = q.loc[q.greater_is_sig], 
                      hue_order = order,
                      order=order,
                      marker='+',
                      size=25/2, 
                      ax=ax,
                      palette=global_palette, 
                      dodge=True, 
                      color='black',
                      linewidth=3.5/2
                       )

        sns.swarmplot(x='Group', 
                      y='greater_y_val', 
                      data = q.loc[q.greater_is_sig], 
                      hue_order = order,
                      order=order,
                      marker='x',
                      size=17.5/2,
                      ax=ax,
                      palette=global_palette, 
                      dodge=True, 
                      color='black',
                      linewidth=3.5/2,
                      edgecolor="black"
                   )

    if sum(q.lower_is_sig > 0):
        sns.swarmplot(x='Group', 
                      y='lower_y_val',
                      data = q.loc[q.lower_is_sig], 
                      hue_order = order,
                      order=order,
                      marker='+',
                      size=25/2, 
                      ax=ax,
                     palette=global_palette, 
                      dodge=True, 
                      color='black',
                      linewidth=3.5/2
                      )

        sns.swarmplot(x='Group', 
                      y='lower_y_val',
                      data = q.loc[q.lower_is_sig], 
                      hue_order = order,
                      order=order,
                      marker='x',
                      size=17.5/2,
                      ax=ax,
                      palette=global_palette, 
                      dodge=True, 
                      color='black',
                      linewidth=3.5/2,
                      edgecolor="black"
                      )

    # Put the legend out of the figure
    plt.legend(bbox_to_anchor=(1.01, .7), loc=2, borderaxespad=0.)
    ax.set_title(None)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    plt.ylim(-.05, 1.05)
    hide_axes=True
    if hide_axes:
        ax.get_legend().remove()
        ax.set_xticks([])
        ax.set(yticks=np.linspace(.1,.9,5), 
               yticklabels=[])
        plt.xlabel(None)
        plt.ylabel(None)
        plt.title(None)

    plt.savefig('../results/Metabolites/Metab-auroc-violinplot-linear-no-axes-pdf', 
                dpi=900,
                bbox_inches='tight', 
                format='pdf')


    # In[6]:


    tmp_palette = {'Melonnpann':'red',
                   'DEBIAS-M Single-task':'lightblue',
                   'DEBIAS-M Multitask':'blue'}


    # In[10]:


    data = pd.read_csv('../results/Metabolites/full-multitask-eval.csv', index_col=0)
    data = pd.DataFrame({'Group': flatten([ [a]*data.shape[0] for a in data.columns ]),
                  'auROC': flatten([data[a].values for a in data.columns])
                 } )

    order = ['Melonnpann', 'DEBIAS-M Single-task', 'DEBIAS-M Multitask']

    plt.subplots(1,
                 figsize=(11.02, 10),
                 dpi=500)

    ax=sns.boxplot(x='Group', 
                      y='auROC', 
                      data=data, 
                      order=order,
                      hue_order = order,
                      palette=tmp_palette,
                      width=.75,
                      fliersize=0
                      )

    handles, labels = ax.get_legend_handles_labels()

    sns.stripplot(x='Group', 
                  y='auROC', 
                  data=data,
                  order=order,
                  hue_order = order,
                  dodge=True,
                  ax=ax, 
                  size=10, 
                  color='k',
                  alpha=.1
               )

    # plot vs scrub significance asterisks

    # 'worse/better than' test
    q=data[['Group']].drop_duplicates()
    q['sig_v_sbcmp_greater'] = q.apply(
                            lambda row: [ wilcoxon( 
                               data.loc[ (data.Group==row.Group) ]['auROC'].values, 
                               data.loc[ (data.Group=='DEBIAS-M Multitask' ) ]['auROC'].values, 
                    alternative='less').pvalue
                                    if row.Group!='DEBIAS-M Multitask' else 1][0],
                       axis=1)

    q['sig_v_sbcmp_less'] = q.apply(
                            lambda row: [ wilcoxon( 
                               data.loc[ (data.Group==row.Group) ]['auROC'].values, 
                               data.loc[ (data.Group=='DEBIAS-M Multitask' ) ]['auROC'].values, 
                    alternative='greater').pvalue
                                    if row.Group!='DEBIAS-M Multitask' else 1][0],
                       axis=1)



    q['lower_y_val'] = 0.275 # data.auROC.min() - 0.03
    q['lower_is_sig'] =  q.sig_v_sbcmp_greater < 0.05

    q['greater_y_val'] = 1.025 #data.auROC.max() + 0.03
    q['greater_is_sig'] =  q.sig_v_sbcmp_less < 0.05

    if sum(q.greater_is_sig > 0):
        sns.swarmplot(x='Group', 
                      y='greater_y_val', 
                      data = q.loc[q.greater_is_sig], 
                      hue_order = order,
                      order=order,
                      marker='+',
                      size=25/2, 
                      ax=ax,
                      palette=tmp_palette, 
                      dodge=True, 
                      color='black',
                      linewidth=3.5/2
                       )

        sns.swarmplot(x='Group', 
                      y='greater_y_val', 
                      data = q.loc[q.greater_is_sig], 
                      hue_order = order,
                      order=order,
                      marker='x',
                      size=17.5/2,
                      ax=ax,
                      palette=tmp_palette, 
                      dodge=True, 
                      color='black',
                      linewidth=3.5/2,
                      edgecolor="black"
                   )

    if sum(q.lower_is_sig > 0):
        sns.swarmplot(x='Group', 
                      y='lower_y_val',
                      data = q.loc[q.lower_is_sig], 
                      hue_order = order,
                      order=order,
                      marker='+',
                      size=25/2, 
                      ax=ax,
                     palette=tmp_palette, 
                      dodge=True, 
                      color='black',
                      linewidth=3.5/2
                      )

        sns.swarmplot(x='Group', 
                      y='lower_y_val',
                      data = q.loc[q.lower_is_sig], 
                      hue_order = order,
                      order=order,
                      marker='x',
                      size=17.5/2,
                      ax=ax,
                      palette=tmp_palette, 
                      dodge=True, 
                      color='black',
                      linewidth=3.5/2,
                      edgecolor="black"
                      )

    # Put the legend out of the figure
    plt.legend(bbox_to_anchor=(1.01, .7), loc=2, borderaxespad=0.)
    ax.set_title(None)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    # plt.ylim(-.05, 1.05)
    plt.ylim(.25, 1.05)
    if hide_axes:
        ax.get_legend().remove()
        ax.set_xticks([])
        ax.set(#yticks=np.linspace(.1,.9,5), 
               yticks=[.3, .5, .7, .9],
               yticklabels=[])
        plt.xlabel(None)
        plt.ylabel(None)
        plt.title(None)
        plt.savefig('../results/Metabolites/full-multitask-eval-no-axes.pdf',
                    dpi=900,
                    bbox_inches='tight', 
                    format='pdf')
        
if __name__=='__main__':
    main()

