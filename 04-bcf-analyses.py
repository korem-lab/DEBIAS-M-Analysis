#!/usr/bin/env python
# coding: utf-8


from General_functions import data_loading, plotting
import pandas as pd
from debiasm.sklearn_functions import AdaptationDebiasMClassifier
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

def main():
    df, md = data_loading.load_HIVRC('../data/HIVRC/')
    df_with_batch = pd.concat([pd.Series( pd.Categorical(md.Study).codes, 
                                         index=md.index), df], axis=1)    
    y=md.label


    np.random.seed(0)
    torch.manual_seed(0)
    dmc = AdaptationDebiasMClassifier()
    dmc.fit(df_with_batch.values, y.values)



    kit_comps = pd.read_csv('../data/HIVRC/DEBIAS-M cross-study metadata - Sheet300.csv', 
                            index_col=0)

    study_info = pd.concat( [ md[['Study']], df_with_batch[[0]] ], axis=1 ).drop_duplicates()                        .sort_values(0).set_index('Study')                        .merge(kit_comps, left_index=True, right_index=True )

    mod_ = dmc.model

    tmp_md = study_info.merge(md, 
                     left_index=True, 
                     right_on='Study').loc[df.index].drop(0, axis=1)

    tmp_md['DNA_extraction_kit'] = tmp_md['ExtractionKit_grouped3']


    tmp_mpp = pd.concat([df_with_batch[0], 
                         tmp_md], axis=1).drop_duplicates()


    vals = mod_.batch_weights[:, 
                      ( ( mod_.batch_weights != 0 
                            ).sum(dim=0) >= mod_.batch_weights.shape[0]*0.75 ) ].detach().numpy()

    bias_comps = pd.DataFrame( 
        vals,
        columns=df_with_batch.columns[1:].values[ 
                      ( ( mod_.batch_weights != 0 
                            ).sum(dim=0)>= mod_.batch_weights.shape[0]*0.75 ) ], 
        index = tmp_mpp[[0, 'DNA_extraction_kit']].drop_duplicates()\
                    .sort_values(0).DNA_extraction_kit
            )

    bcm=bias_comps.reset_index().fillna('Unknown').groupby('DNA_extraction_kit')[bias_comps.columns
                                                                                ].mean()




    stain_label = pd.read_csv('../data/HIVRC/full_gram_stain.csv', index_col=0
                                     )[['species', 'stain']].drop_duplicates()
    vvv= stain_label.groupby('species').nunique()
    stain_label = stain_label.loc[stain_label.species.isin(vvv.loc[vvv.stain==2].index.str.lower())==False]
    stain_label['Genus'] = stain_label['species'].str.lower()
    stain_label['Gram Stain'] = stain_label['stain'].fillna('Not applicable')


    bcm.columns = bcm.columns.str.lstrip('merged.otu:').str.lower().str.split(':').str[0]        .str.replace('_', ' ').str.lower()



    fff=bcm.T.reset_index()[['index']]        .merge(stain_label[['Genus', 'Gram Stain']], 
                  left_on='index', right_on='Genus')

    fff = fff.loc[fff['Gram Stain']!= 'Not applicable'].drop_duplicates()


    qqs=[]

    for a in bcm.index.values:
        tmp = bcm.loc[a].reset_index().merge(fff).iloc[:, 1:]
        tmp.columns =['W', 'Genus', 'Gram Stain']
        tmp['Kit']=a

        qqs.append(tmp.copy())


    qqq=pd.concat(qqs)



    plt.figure(figsize=(8,8))
    ax = sns.boxplot(x = 'Kit', 
                  y = 'W',
                  hue = 'Gram Stain',
                  data = qqq.loc[(qqq.W!=0)&(qqq.Kit.isin(['MoBio', 
                                     'QIAamp', 
                                      'MagNA Pure'
                                     ]))], 
                     fliersize=0
                 )


    sns.swarmplot(x = 'Kit', 
                  y = 'W',
                  hue = 'Gram Stain',
                  data = qqq.loc[(qqq.W!=0)&(qqq.Kit.isin(['MoBio', 
                                     'QIAamp', 
                                      'MagNA Pure'
                                     ]))], 
                  dodge=True, 
                  color='k',
                  s=5,#10
                 )

    handles, labels = ax.get_legend_handles_labels()
    l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig('../results/HIVRC/gram-stain.pdf', 
                dpi=900, 
                bbox_inches='tight', 
                format='pdf')




    for kit in ['MoBio', 'QIAamp', 'MagNA Pure']:
        print(kit)
        tmp=qqq.loc[qqq.Kit==kit]
        tmp=tmp.loc[tmp.W!=0]
        print(mannwhitneyu(tmp.loc[tmp['Gram Stain']=='Gram+'].W, 
                          tmp.loc[tmp['Gram Stain']=='Gram-'].W))


    plt.figure(figsize=(8,8))
    ax = sns.boxplot(x = 'Kit', 
                  y = 'W',
                  hue = 'Gram Stain',
                  data = qqq.loc[(qqq.W!=0)&(qqq.Kit.isin(['MoBio', 
                                     'QIAamp', 
                                      'MagNA Pure'
                                     ]))], 
                     fliersize=0
                 )


    sns.swarmplot(x = 'Kit', 
                  y = 'W',
                  hue = 'Gram Stain',
                  data = qqq.loc[(qqq.W!=0)&(qqq.Kit.isin(['MoBio', 
                                     'QIAamp', 
                                      'MagNA Pure'
                                     ]))], 
                  dodge=True, 
                  color='k',
                  s=5,#10
                 )

    ax.legend().remove()
    ax.set(yticklabels=[], xticklabels=[])
    plt.xlabel(None)
    plt.ylabel(None)
    plt.savefig('../results/HIVRC/gram-stain-no-axes.pdf', 
                dpi=900, 
                bbox_inches='tight', 
                format='pdf')




    kit_comps = pd.read_csv('../data/HIVRC/DEBIAS-M cross-study metadata - Sheet300.csv', 
                            index_col=0)

    study_info = pd.concat( [ md[['Study']], df_with_batch[[0]] ], axis=1 ).drop_duplicates()                        .sort_values(0).set_index('Study')                        .merge(kit_comps, left_index=True, right_index=True )



    study_info.drop(0, axis=1).to_csv('../results/HIVRC/processed_hiv_metadata.csv')


    pd.DataFrame( dmc.model.batch_weights.detach().numpy()[study_info[0].values], 
                  index=study_info.index ).to_csv('../results/HIVRC/processed_hiv_bcfs.csv')


    rbm_info=pd.read_csv('../data/HIVRC/extraction_protocols.tsv', sep='\t', index_col=0).sort_index()
    rbm_info.index=study_info.index
    inds = rbm_info['Robot / manual'].isna()==False
    print( mannwhitneyu( np.std( dmc.model.batch_weights.detach().numpy()[study_info[0].values[inds]], axis=1 )[
                         rbm_info['Robot / manual'].loc[inds]=='Manual'], 
                 np.std( dmc.model.batch_weights.detach().numpy()[study_info[0].values[inds]], axis=1 )[
                         rbm_info['Robot / manual'].loc[inds]=='Robot'], 
                 alternative='less'
                ) )


    plt.figure(figsize=(8,8))
    ax = sns.boxplot( y = np.std( dmc.model.batch_weights.detach().numpy()[study_info[0].values[inds]], axis=1 ), 
                 x = rbm_info['Robot / manual'].loc[inds]
               )

    sns.swarmplot(y = np.std( dmc.model.batch_weights.detach().numpy()[study_info[0].values[inds]], axis=1 ), 
                  x = rbm_info['Robot / manual'].loc[inds], 
                  color = 'black', 
                  ax = ax, 
                  s=10
               )
    plt.ylabel('Standard deviation of BCFs')
    plt.xlabel(None)

    plt.savefig('../results/HIVRC/bcf-manual-v-robot-with-axes.pdf', 
                dpi=900, 
                format='pdf', 
                bbox_inches='tight')


    plt.figure(figsize=(8,8))
    ax = sns.boxplot( y = np.std( dmc.model.batch_weights.detach().numpy()[study_info[0].values[inds]], axis=1 ), 
                 x = rbm_info['Robot / manual'].loc[inds]
               )

    sns.swarmplot(y = np.std( dmc.model.batch_weights.detach().numpy()[study_info[0].values[inds]], axis=1 ), 
                  x = rbm_info['Robot / manual'].loc[inds], 
                  color = 'black', 
                  ax = ax, 
                  s=10
               )
    plt.ylabel(None)
    plt.yticks(ticks = ax.get_yticks(), 
               labels=[])
    plt.xticks(ticks=ax.get_xticks(), labels=[])
    plt.xlabel(None)

    plt.savefig('../results/HIVRC/bcf-manual-v-robot-no-axes.pdf', 
                dpi=900, 
                format='pdf', 
                bbox_inches='tight')

    inds = study_info['ExtractionKit_grouped3'].isna()==False


    pc=PCA(n_components=2)
    xx = pc.fit_transform( dmc.model.batch_weights.detach().numpy()[study_info[0].values[inds]] )

    study_info['Extraction kit'] = study_info['ExtractionKit_grouped3']#.str.rstrip('[ - other]')
    eee=study_info['Extraction kit'].value_counts() < 2

    eee.loc[eee]
    study_info.loc[ study_info['Extraction kit'].isin(eee.loc[eee].index), 
                   'Extraction kit' ] = 'Other'
    study_info['Region'] = study_info['region']

    plt.figure(figsize=(8,8))
    ax = sns.scatterplot(xx[:, 0],
                         xx[:, 1],
                          hue = study_info['Extraction kit'].fillna('Unknown').loc[inds],
                         style = study_info['Region'].fillna('Unknown').loc[inds],
                         s = 500
                        )


    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, 
              markerscale=3.5)
    ax.set(yticklabels=[], xticklabels=[])


    plt.savefig('../results/HIVRC/kit-region-bcf-pca.pdf', 
                format='pdf', 
                dpi=900,
                bbox_inches='tight'
               )
    plt.show()



    study_info['Extraction kit'] = study_info['ExtractionKit_grouped3']
    inds = study_info['Extraction kit'].isna()==False

    pc=PCA(n_components=2)
    md_ww = dmc.model.batch_weights.detach().numpy()[study_info[0].values[inds]]
    md_ww[md_ww==0]=md_ww.max()
    xx = pc.fit_transform( md_ww )

    study_info['Extraction kit'] = study_info['ExtractionKit_grouped3']#.str.rstrip('[ - other]')
    eee=study_info['Extraction kit'].value_counts() < 2
    eee.loc[eee]
    study_info.loc[ study_info['Extraction kit'].isin(eee.loc[eee].index), 
                   'Extraction kit' ] = 'Other'
    study_info['Region'] = study_info['region']

    plt.figure(figsize=(8,8))
    ax = sns.scatterplot(xx[:, 0],
                         xx[:, 1],
                         hue = study_info['Extraction kit'].fillna('Unknown').loc[inds],
                         style = study_info['Region'].fillna('Unknown').loc[inds],
                         s = 500
                        )


    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, 
              markerscale=3.5)
    ax.set(yticklabels=[], xticklabels=[])


    plt.savefig('../results/HIVRC/kit-region-bcf-pca-emax.pdf', 
                format='pdf', 
                dpi=900,
                bbox_inches='tight'
               )
    plt.show()


    # ## Copy number



    rrndb_mean_cs = pd.read_csv('../data/HIVRC/rrnDB-5.8_pantaxa_stats_NCBI.tsv', sep='\t')
    rrndb_mean_cs = rrndb_mean_cs.loc[rrndb_mean_cs['rank']=='genus'].set_index('name')[['mean']]
    rrndb_mean_cs.index = rrndb_mean_cs.index.str.lower().str.split(' ').str[0]
    rrndb_mean_cs.columns = ['Copy number']

    qq=bcm.copy().reset_index()
    qq.columns = ['dna kit'] + list( bias_comps.columns.str.split('_').str[0].str.lower() )
    qqq=pd.concat(qqs)

    qqq['merge_col']=qqq.Genus.str.split(' ').str[0]
    qqq= qqq.merge(rrndb_mean_cs, 
                   left_on='merge_col',
                   right_index=True
                   )

    tmp = qq.T.merge(rrndb_mean_cs, left_index=True, right_index=True)

    qqq.loc[qqq.W==0, 'W'] = qqq.W.max()

    qqq = qqq.groupby( list(qqq.columns[2:-1].values) )[['W', 'Copy number']]\
                        .agg(lambda x: x.mean() ).reset_index()



    plot_dat=qqq.copy()
    plt.figure(figsize=(8,8))
    ax=sns.scatterplot(x = 'Copy number', 
                       y = 'W', 
                       data=plot_dat, 
                       s=100, 
                       alpha=.5
                  )
    plot_dat_ = plot_dat.copy()
    lr=LinearRegression()
    lr.fit(plot_dat_[['Copy number']], getattr(plot_dat_, 'W')  )
    sns.lineplot( y = lr.predict( plot_dat_[['Copy number']].reset_index(drop=True) ), 
                  x = plot_dat_['Copy number'].reset_index(drop=True), 
                linewidth=5, 
                color='blue', 
                )
    print(pearsonr(plot_dat_['Copy number'].values, plot_dat_.W.values))
    plt.savefig('../results/HIVRC/copy-number-bcf-emax.pdf', 
                format='pdf', 
                dpi=900,
                bbox_inches='tight'
               )
    plt.show()



    plot_dat=qqq.copy()
    plt.figure(figsize=(8,8))
    ax=sns.scatterplot(x = 'Copy number', 
                       y = 'W', 
                       data=plot_dat, 
                       s=100, 
                       alpha=.5
                  )
    plot_dat_ = plot_dat.copy()
    lr=LinearRegression()
    lr.fit(plot_dat_[['Copy number']], getattr(plot_dat_, 'W')  )
    sns.lineplot( y = lr.predict( plot_dat_[['Copy number']].reset_index(drop=True) ), 
                  x = plot_dat_['Copy number'].reset_index(drop=True), 
                linewidth=5, 
                color='blue', 
                )

    plt.ylabel(None)
    plt.xlabel(None)
    ax.set(yticklabels=[], 
           xticklabels=[]
           )
    print(pearsonr(plot_dat_['Copy number'].values, plot_dat_.W.values))
    plt.savefig('../results/HIVRC/copy-number-bcf-emax-no-axes.pdf', 
                format='pdf', 
                dpi=900,
                bbox_inches='tight'
               )
    plt.show()


if __name__=='__main__':
    main()
