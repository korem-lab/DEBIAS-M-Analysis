import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from scipy.stats import wilcoxon
from General_functions.plotting import global_palette
global_palette['Baseline'] = global_palette['linear']
global_palette['DEBIAS-M'] = global_palette['debias-m']


def main():
    result_paths = glob.glob('../data/all-simulation-runs/*.csv')
    dfs = pd.concat( [ pd.read_csv(a, index_col=0) for a in result_paths ] )
    dfs = dfs.loc[dfs.Pheno_noise > 0]
    cols_of_interest = dfs.columns[2:-2]
    default_vals = pd.DataFrame({'N_studies':4, 
                                 'Pheno_noise':0.1, 
                                 'Read_depth':1e5, 
                                 'N_features':1e3, 
                                 'N_per_study':96
                                 }, 
                                index=[0]
                               )

    for hide_axes in [False, True]:
        for col_of_interest in default_vals.columns:

            df_tmp = dfs.merge(default_vals.drop(col_of_interest, axis=1))

            df_tmp = df_tmp.groupby( list( df_tmp.columns.drop('JSD').values) 
                                       )['JSD'].median().reset_index()

            plt.figure(figsize=(12, 12))
            ax=sns.boxplot(hue='Group', 
                        x = col_of_interest,
                        y = 'JSD', 
                        data=df_tmp, 
                        palette=global_palette,
                        fliersize=0)

            sns.stripplot(hue ='Group', 
                          x = col_of_interest,
                          y = 'JSD', 
                          data=df_tmp, 
                          color='black',
                          s=10, 
                          ax=ax, 
                          dodge=True
                          )




            q = pd.DataFrame( {'sig_v_debias':  [ wilcoxon( df_tmp.loc[( df_tmp[col_of_interest] == a)&
                                         ( df_tmp['Group'] == 'Baseline' )].JSD, 
                                df_tmp.loc[( df_tmp[col_of_interest] == a)&
                                           ( df_tmp['Group'] == 'DEBIAS-M' )].JSD, 
                                         alternative='greater'
                              ).pvalue < 1e-3
                               for a in df_tmp[col_of_interest].unique() ], 
                             col_of_interest:df_tmp[col_of_interest].unique(), 
                           'Group':['Baseline'] * df_tmp[col_of_interest].nunique()
                                }
                        )


            print(pd.DataFrame( {'sig_v_debias':  [ wilcoxon( df_tmp.loc[( df_tmp[col_of_interest] == a)&
                                         ( df_tmp['Group'] == 'Baseline' )].JSD, 
                                df_tmp.loc[( df_tmp[col_of_interest] == a)&
                                           ( df_tmp['Group'] == 'DEBIAS-M' )].JSD, 
                                         alternative='greater'
                              ).pvalue
                               for a in df_tmp[col_of_interest].unique() ], 
                             col_of_interest:df_tmp[col_of_interest].unique(), 
                           'Group':['Baseline'] * df_tmp[col_of_interest].nunique()
                                }
                        ))

            q['lower_y_val'] = 0.19

            if sum(q.sig_v_debias > 0):
                sns.swarmplot(
                              hue='Group', 
                              x=col_of_interest,
                              y='lower_y_val',
                              data = q.loc[q.sig_v_debias], 
                              marker='+',
                              size=25/2, 
                              ax=ax,
                             palette=global_palette, 
                              dodge=True, 
                              color='black',
                              linewidth=3.5/2, 
                              )

                sns.swarmplot(hue='Group', 
                              x=col_of_interest,
                              y='lower_y_val',
                              data = q.loc[q.sig_v_debias], 
                              marker='x',
                              size=17.5/2,
                              ax=ax,
                              palette=global_palette, 
                              dodge=True, 
                              color='black',
                              linewidth=3.5/2,
                              edgecolor="black", 
                              )

            plt.ylabel('JSD')
            plt.legend().remove()
            plt.ylim(0, 0.2)
            ax.set(yticks=np.linspace(0, 0.2, 5),
                  )

            out_path = '../results/Simulations/JSD-plot-{}.pdf'.format(col_of_interest)
            if hide_axes:
                ax.set_xticks([])
                ax.set(yticks=np.linspace(0, 0.2, 5),
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
                
                
if __name__=='__main__':
    main()      
                