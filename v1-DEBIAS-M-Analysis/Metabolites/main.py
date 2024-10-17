

import numpy as np
import pandas as pd
from . import metab_prediction
from General_functions.plotting import *
import torch

def produce_auroc_violinplot( data, 
                              out_path,
                              order=['linear', 
                                     'combat', 
                                     'conqur', 
                                     'debias-m'],
                              hide_axes=False,
                              global_palette=global_palette
                            ):

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
                  alpha=0.1
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



    q['lower_y_val']=data.auROC.min() - 0.03
    q['lower_is_sig'] =  q.sig_v_sbcmp_greater < 0.05

    q['greater_y_val']=data.auROC.max() + 0.03
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

#     data.to_csv(out_path.replace('.pdf', '.csv'))
        
    sns.violinplot(x='Group', 
                y='auROC', 
                data=data, 
                order=order,
                hue_order = order,
                palette=global_palette,
                width=.75,
                fliersize=0, 
                ax=ax
                )


    # Put the legend out of the figure
    plt.legend(bbox_to_anchor=(1.01, .7), loc=2, borderaxespad=0.)
    ax.set_title(None)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    
    if hide_axes:
        ax.get_legend().remove()
        ax.set_xticks([])
        ax.set(yticklabels=[])
        plt.xlabel(None)
        plt.ylabel(None)
        plt.title(None)
    

    plt.savefig(out_path, 
                    dpi=900,
                   bbox_inches='tight', 
                   format='pdf')

def Metabolites_functions(md, df_with_batch, conq_out, comb_out, seed=0):

    np.random.seed(seed)
    torch.manual_seed(seed)

    metab_prediction.run_metab_predictions(md,
                                           df_with_batch, 
                                           conq_out, 
                                           comb_out
                                          )
    
    data=pd.read_csv('../results/Metabolites/Metab-auroc-boxplot-linear.csv', index_col=0)
    produce_auroc_violinplot(data, 
                             out_path = '../results/Metabolites/Metab-auroc-violinplot-linear.pdf'
                            )
    
    
    metab_prediction.run_melonpann_and_multitask_benchmarks(df_with_batch, md)
    return(None)















