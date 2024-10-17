
import sys
sys.path.append('../v1-DEBIAS-M-Analysis/')

from General_functions import data_loading, plotting
import pandas as pd
from debiasm.sklearn_functions import AdaptationDebiasMClassifier, DebiasMClassifier
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from General_functions import plotting
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from debiasm.torch_functions import rescale
from sklearn.metrics import pairwise_distances
from General_functions.plotting import global_palette


def get_cross_sample_jaccard(df, 
                             df_pseudo_with_batch, 
                             df_tmp_pseudo, 
                             df_tmp_no_pseudo, 
                             thresh=1e-4
                             ):
                             
    met='jaccard'

    diff_study_mask =  pairwise_distances( df_pseudo_with_batch.iloc[:, 0:1], 
                                 metric='manhattan') > 0

    diff_study_mask=np.triu(diff_study_mask)
    diff_study_mask = pd.DataFrame( diff_study_mask, 
                            index=df_pseudo_with_batch.iloc[:, 0], 
                            columns=df_pseudo_with_batch.iloc[:, 0]
                            )

    study_pairs = diff_study_mask.index.astype(str)[np.newaxis, :] + ' - ' + \
                diff_study_mask.columns.astype(str)[:, np.newaxis]


    def get_mean_per_study_dists(dists, study_pairs=study_pairs, mask=diff_study_mask):
        return( 
            pd.DataFrame({'distances':dists[mask], 
                          'study_pairs':study_pairs[mask]})\
                    .groupby('study_pairs')['distances'].mean().values
        )


    base_dists_no_pseudo = get_mean_per_study_dists(pairwise_distances( rescale(df).values > thresh , 
                                                          metric=met) )


    base_dists = get_mean_per_study_dists( 
                    pairwise_distances( rescale(1+df).values > thresh, 
                                              metric=met) )

    pseudo_dists = get_mean_per_study_dists(
                    pairwise_distances( rescale(df_tmp_pseudo.iloc[:, :-1]).values > thresh, 
                                              metric=met) )
    no_pseudo_dists = get_mean_per_study_dists(
                    pairwise_distances( \
                                rescale(df_tmp_no_pseudo.iloc[:, :-1]).values > thresh , 
                                              metric=met) )

    plt.figure(figsize=(4,8))
    ax=sns.boxplot(
                 y = list(base_dists_no_pseudo) + \
                        list(base_dists) + \
                        list(no_pseudo_dists) +\
                            list(pseudo_dists), 
                 x=['Raw samples no pseudocount'] * base_dists_no_pseudo.shape[0] +\
                    ['Raw samples w/ pseudocount'] * base_dists.shape[0] + \
                    ['DEBIAS-M']*no_pseudo_dists.shape[0] +
                         ['pseudocount DEBIAS-M'] * \
                             pseudo_dists.shape[0],
                 palette={'Raw samples no pseudocount':global_palette['linear'], 
                          'Raw samples w/ pseudocount':global_palette['linear'], 
                          'pseudocount DEBIAS-M':'lightblue', 
                          'DEBIAS-M':global_palette['debias-m']
                         }, 
                 )

    plt.xticks(rotation=90)
    return(ax)



def get_cross_sample_braycurt(df, 
                             df_pseudo_with_batch, 
                             df_tmp_pseudo, 
                             df_tmp_no_pseudo, 
                             thresh=1e-4
                             ):
                             


    met='braycurtis'

    diff_study_mask =  pairwise_distances( df_pseudo_with_batch.iloc[:, 0:1], 
                                 metric='manhattan') > 0

    diff_study_mask=np.triu(diff_study_mask)
    diff_study_mask = pd.DataFrame( diff_study_mask, 
                            index=df_pseudo_with_batch.iloc[:, 0], 
                            columns=df_pseudo_with_batch.iloc[:, 0]
                            )

    study_pairs = diff_study_mask.index.astype(str)[np.newaxis, :] + ' - ' + \
                diff_study_mask.columns.astype(str)[:, np.newaxis]


    def get_mean_per_study_dists(dists, study_pairs=study_pairs, mask=diff_study_mask):
        return( 
            pd.DataFrame({'distances':dists[mask], 
                          'study_pairs':study_pairs[mask]})\
                    .groupby('study_pairs')['distances'].mean().values
        )

    base_dists_no_pseudo = get_mean_per_study_dists(pairwise_distances( rescale(df).values , 
                                                          metric=met) )


    base_dists = get_mean_per_study_dists( 
                    pairwise_distances( rescale(1+df).values, 
                                              metric=met) )

    pseudo_dists = get_mean_per_study_dists(
                    pairwise_distances( rescale(df_tmp_pseudo.iloc[:, :-1]).values , 
                                              metric=met) )
    no_pseudo_dists = get_mean_per_study_dists(
                    pairwise_distances( \
                                rescale(df_tmp_no_pseudo.iloc[:, :-1]).values , 
                                              metric=met) )

    plt.figure(figsize=(4,8))
    ax=sns.boxplot(
                 y = list(base_dists_no_pseudo) + \
                        list(base_dists) + \
                        list(no_pseudo_dists) +\
                            list(pseudo_dists), 
                 x=['Raw samples no pseudocount'] * base_dists_no_pseudo.shape[0] +\
                    ['Raw samples w/ pseudocount'] * base_dists.shape[0] + \
                    ['DEBIAS-M']*no_pseudo_dists.shape[0] +
                         ['pseudocount DEBIAS-M'] * \
                             pseudo_dists.shape[0],
                 palette={'Raw samples no pseudocount':global_palette['linear'], 
                          'Raw samples w/ pseudocount':global_palette['linear'], 
                          'pseudocount DEBIAS-M':'lightblue', 
                          'DEBIAS-M':global_palette['debias-m']
                         }, 
                 )



    # plt.yticks([0, .4, .8, 1.2])
    plt.xticks(rotation=90)
    return(ax)



def main(seed=123):
    df, md = data_loading.load_HIVRC('../data/HIVRC/')
    df_with_batch = pd.concat([pd.Series( pd.Categorical(md.Study).codes, 
                                         index=md.index), df], axis=1)    
    y=md.label
    
    
    
    ### DEBIAS with pseudocount -- adaptation to use all studies/labels 
                       # since we're just comparing sample compositions (no label-based benchmark here)
    np.random.seed(seed)
    torch.manual_seed(seed)

    df_with_batch_pseudocount = pd.concat([pd.Series( pd.Categorical(md.Study).codes, 
                                         index=md.index), 1+df], axis=1) 

    dmc_pseudocount = AdaptationDebiasMClassifier()
    dmc_pseudocount.fit(df_with_batch_pseudocount.values, y.values)

    kit_comps = pd.read_csv('../data/HIVRC/DEBIAS-M cross-study metadata - Sheet300.csv', 
                            index_col=0)

    study_info = pd.concat( [ md[['Study']], 
                             df_with_batch_pseudocount[[0]] ], 
                           axis=1 
                          ).drop_duplicates()\
                        .sort_values(0).set_index('Study')\
                    .merge(kit_comps, left_index=True, right_index=True )

    mod_pseudocount = dmc_pseudocount.model

    df_pseudo_out=dmc_pseudocount.transform(df_with_batch_pseudocount)

    tmp_md = study_info.merge(md, 
                     left_index=True, 
                     right_on='Study').loc[df.index].drop(0, axis=1)

    df_tmp=rescale( 1 + df )
    df_tmp['Study']= tmp_md.loc[df.index]['region'] + ' - ' + md['Study']

    df_tmp_pseudo=rescale( df_pseudo_out.copy() )
    df_tmp_pseudo['Study']= tmp_md.loc[df_tmp_pseudo.index]['region'] + ' - ' + md['Study']

    ## post-hoc removal of pseudocount debias outputs below a certain threshold
    thresh=1e-4
    df_ttt=df_tmp_pseudo.values
    df_ttt[:, :-1][df_ttt[:, :-1]<thresh]=0
    df_ttt[:, :-1]=rescale(df_ttt[:, :-1])
    df_tmp_pseudo=pd.DataFrame(df_ttt, 
                               columns=df_tmp_pseudo.columns, 
                               index=df_tmp_pseudo.index)
    
    
    
    ### DEBIAS without pseudocount
    np.random.seed(seed)
    torch.manual_seed(seed)

    df_with_batch_no_pseudocount = pd.concat([pd.Series( pd.Categorical(md.Study).codes, 
                                         index=md.index), df], axis=1) 

    dmc_no_pseudocount = AdaptationDebiasMClassifier()
    dmc_no_pseudocount.fit(df_with_batch_no_pseudocount.values, y.values)

    kit_comps = pd.read_csv('../data/HIVRC/DEBIAS-M cross-study metadata - Sheet300.csv', 
                            index_col=0)

    study_info = pd.concat( [ md[['Study']], 
                             df_with_batch_no_pseudocount[[0]] ], 
                           axis=1 
                          ).drop_duplicates()\
                        .sort_values(0).set_index('Study')\
                    .merge(kit_comps, left_index=True, right_index=True )

    mod_no_pseudocount = dmc_no_pseudocount.model

    df_no_pseudo_out=dmc_no_pseudocount.transform(df_with_batch_no_pseudocount)

    tmp_md = study_info.merge(md, 
                     left_index=True, 
                     right_on='Study').loc[df.index].drop(0, axis=1)

    df_tmp_no_pseudo=rescale( df_no_pseudo_out.copy() )
    df_tmp_no_pseudo['Study']= tmp_md.loc[df_tmp_no_pseudo.index]['region'] + ' - ' + md['Study']

    
    get_cross_sample_jaccard(df, 
                             df_with_batch_pseudocount, 
                             df_tmp_pseudo, 
                             df_tmp_no_pseudo, 
                             thresh=thresh
                             )
    plt.savefig('../plots/pseudocount-evaluations/cross-study-presence-jaccard.pdf', 
                                    format='pdf', 
                                    dpi=900, 
                                    bbox_inches='tight'
                                    )

    get_cross_sample_braycurt(df, 
                             df_with_batch_pseudocount, 
                             df_tmp_pseudo, 
                             df_tmp_no_pseudo
                             )
    
    plt.savefig('../plots/pseudocount-evaluations/cross-study-braycurtis.pdf', 
                                    format='pdf', 
                                    dpi=900, 
                                    bbox_inches='tight'
                                    )

    print('major success!!')
    return(None)


if __name__=='__main__':
    main()
    
    
    
    