import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
# from matplotlib_venn import venn3_unweighted
import matplotlib
import seaborn as sns
import os
import scipy
from scipy.stats import wilcoxon


## plotting parameters
sns.set_theme(font_scale=2)

sns.set(rc={'axes.facecolor':'white', 
            'figure.facecolor':'white', 
            'axes.edgecolor':'black', 
            'grid.color': 'black'
            }, 
       font_scale=2)

global_palette = {'linear':'#F97306', 
                  'debias-m':'#069AF3', 
                  'conqur':'red', 
                  'combat':'#15B01A', 
                  'snm':'pink'}


def produce_auroc_boxplot(data, 
                          out_path,
                          order=['linear', 
                                 'combat', 
                                 'conqur', 
                                 'snm',
                                 'debias-m'],
                          hide_axes=True,
                          global_palette=global_palette, 
                          sig_dict={}
                        ):

    order=['linear', 
           'combat', 
           'conqur', 
           'snm',
           'debias-m']
    
    if 'snm' not in data.Group.unique():
        order = ['linear', 
                 'combat', 
                 'conqur', 
                 'debias-m'
                 ]
        
    data.to_csv(out_path.replace('.pdf', '.csv'))

    plt.subplots(1,
                 figsize=(11.02, 10),
                 dpi=500)

    ax=sns.boxplot(x='Group', 
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
                  color='k'
               )


    q = pd.DataFrame(sig_dict, index=['is_sig']).T.reset_index()
    q.columns = ['Group', 'sig_v_debias']
#     q['lower_y_val'] = -0.025
    
    q['lower_y_val'] = 0.275

    if sum(q.sig_v_debias > 0):
        sns.swarmplot(x='Group', 
                      y='lower_y_val',
                      data = q.loc[q.sig_v_debias], 
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
                      data = q.loc[q.sig_v_debias], 
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
    sns.boxplot(x='Group', 
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
#     plt.ylim(-.05, 1.05)
    plt.ylim(0.25, 1.05)
    if hide_axes:
        ax.get_legend().remove()
        ax.set_xticks([])
        ax.set(#yticks=np.linspace(.1,.9,5),
               yticks=[0.3,0.5,0.7, 0.9],
               yticklabels=[])
        plt.xlabel(None)
        plt.ylabel(None)
        plt.title(None)
        plt.savefig(out_path[:-4] + '-no-axes'+out_path[-4:], 
                    dpi=900,
                    bbox_inches='tight', 
                    format='pdf')
        
    else:
        plt.savefig(out_path, 
                    dpi=900,
                   bbox_inches='tight', 
                   format='pdf')
        
    return(None)
        
    