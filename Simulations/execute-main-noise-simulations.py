import os
import numpy as np
import pandas as pd
from debiasm.sklearn_functions import AdaptationDebiasMClassifier, \
                                            DebiasMClassifier
from debiasm.torch_functions import rescale
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
import torch
from sklearn.model_selection import cross_val_score
from debiasm.sklearn_functions import batch_weight_feature_and_nbatchpairs_scaling
import subprocess

def experiment_run(samples_,
                   n_studies, 
                   n_per_study,
                   n_features,
                   phenotype_std,
                   read_depth, 
                   smp_noise_lvl=0,
                   bcf_stdv = 2, 
                   linear_weights_stdv = 2,
                   seed=0):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
    # drop samples/features not needed for this run
    samples=samples_.sample(n=n_studies * n_per_study)\
                .T.sample(n=n_features).T
    
    ## simulated bcfs
    true_bcfs = np.random.normal(scale=bcf_stdv,
                                 size = (n_studies, n_features)
                                 )
    
    ## simulated linear weights
    true_weights = np.random.normal(scale=linear_weights_stdv, 
                                    size = n_features
                                    )

    ## simulated phenotypes
    phenotypes_ = samples @ true_weights[:, np.newaxis]
    if phenotype_std > 0:
        phenotypes_ += np.random.normal(scale = phenotype_std,
                                       size=phenotypes_.shape)
    
    studies = pd.Series( [ i // n_per_study for i in range(samples.shape[0]) 
                                     ] , 
                        index=samples.index, 
                        name='studies')

    samples_transformed = rescale(samples.values *\
                                    np.power(2, true_bcfs)[studies] 
                                    )
    
#     samples_transformed = samples_transformed + \
#                                 smp_noise_lvl * ( \
#                                 np.random.rand(*samples_transformed.shape) ) ## range \in [0,1]
    
    samples_transformed = samples_transformed + \
                                ( smp_noise_lvl * ( \
                                  np.random.rand(*samples_transformed.shape) ) *
                                 ( samples_transformed>0 )
                                )
    
    
    samples_transformed[ samples_transformed< 0 ]=0 ## zero out anything <0
    
    samples_transformed = ( rescale(samples_transformed) *read_depth ).astype(int)
    
    samples_transformed = pd.DataFrame(samples_transformed,
                                       index=samples.index, 
                                       columns=samples.columns)
    
    test_set_inds = studies==studies.max()
    train_set_inds = studies!=studies.max()
    
    phenotypes = phenotypes_.loc[train_set_inds]
    df_with_batch = pd.concat([studies, samples_transformed], axis=1 )
    
    dm = DebiasMClassifier(x_val = df_with_batch.loc[test_set_inds].values, 
                           batch_str = batch_weight_feature_and_nbatchpairs_scaling(1e4, 
                                                                                    df_with_batch)
                          )
    dm.fit(X=df_with_batch.loc[train_set_inds].values, 
           y=phenotypes.values[:, 0] > phenotypes.median().values[0]    
          )
    
    
    
    transformed_jsds = jensenshannon( dm.transform(pd.concat([studies,
                                      samples_transformed], 
                                      axis=1 )
                                          ).T,
                              samples.T)

    baseline_jsds = jensenshannon(samples_transformed.T, samples.T)
    n_samples = samples.shape[0]

    summary_df = pd.DataFrame({'Group':['Baseline']*n_samples +\
                                               ['DEBIAS-M']*n_samples,
                               'JSD':list(baseline_jsds**2) +\
                                           list(transformed_jsds**2),
                               'N_studies':n_studies, 
                               'N_per_study':n_per_study, 
                               'N_features':n_features, 
                               'Pheno_noise':phenotype_std, 
                               'Read_depth':read_depth, 
                               'BCF_std':bcf_stdv,
                               'Micro_noise':smp_noise_lvl,
                               'is_test_set':list(test_set_inds.values)*2,
                               'run_num':seed
                              })
    
    run_description = '_'.join( [a.replace('_', '-') + '-' + str(b)
                                 for a,b in 
                                   {'N_studies':n_studies, 
                                   'N_per_study':n_per_study, 
                                   'N_features':n_features, 
                                   'Pheno_noise':phenotype_std, 
                                   'Read_depth':int(read_depth), 
                                   'BCF_std':bcf_stdv,
                                    'Micro_noise':smp_noise_lvl,
                                   'run_num':seed}.items()]
                                )
    
    return(summary_df, 
           run_description, 
           true_bcfs, 
           dm, 
           samples, 
           samples_transformed, 
           phenotypes_.iloc[:, 0] > np.median( phenotypes_.values[:, 0] ), 
           studies, 
           test_set_inds
           )


def main(out_path='all-micro-noise-simulation-results-final.csv'):
    
    if os.path.exists(out_path):
        raise(ValueError('out path file already exists!'))
    
    pths = ['tmp-dataset-with-added-bias/percnorm_out.csv',
            'tmp-dataset-with-added-bias/combat-out.csv',
            'tmp-dataset-with-added-bias/conqur-out.csv',
            'tmp-dataset-with-added-bias/MMUPHin_out.csv',
           ]
    first_run=True
    for seed in range(1,26):
        for n_features in [100]:
            for n_studies in [4]:
                for n_per_study in [96]:
                    for read_depth in [1e4]:
                        for phenotype_std in [.1]:
                            for smp_noise_lvl_ in [0, 0.001,  0.01, 0.05, 0.1]:
                                
                                if  (\
                                    ( n_features == 100 ) + \
                                    ( n_studies == 4 ) + \
                                    ( read_depth == 1e4 ) + \
                                    ( n_per_study == 96 ) + \
                                    ( phenotype_std == .1 ) + \
                                    (smp_noise_lvl_ == 0 )\
                                      )>= 5: 

                                
                                    print(seed)
                                    data = pd.read_csv('../Simulations/generated-100-feat-datasets/dataset-seed-{}.csv'.format(seed), 
                                                      index_col=0)
                                    
                                    if n_features==10:
                                        data = data + 1e-4

                                    summary_df, \
                                       run_description,\
                                       true_bcfs, \
                                       dm, \
                                       samples, \
                                       samples_transformed, \
                                       y, \
                                       studies, \
                                       test_set_inds = experiment_run(data,
                                                                n_studies, 
                                                                n_per_study,
                                                                n_features,
                                                                phenotype_std,
                                                                read_depth, 
                                                                bcf_stdv = 2, 
                                                                seed=seed, 
                                                                smp_noise_lvl=smp_noise_lvl_        
                                                                )
                                    
                                    pd.DataFrame({'label':y, 
                                                 'Study':studies}, 
                                                 index=samples_transformed.index
                                                ).to_csv('tmp-dataset-with-added-bias/md.csv')
                                    samples_transformed.to_csv('tmp-dataset-with-added-bias/data.csv')

                                    ## run competing methods' batch corrections
                                    os.system('Rscript run-simulation-R-BC-methods.R')

                                    all_df=summary_df.copy()
                                    all_df=all_df.loc[all_df.is_test_set]

                                    ## read each methods' output
                                    for a in pths:
                                        tt=pd.read_csv(a, index_col=0)
                                        tt.index=samples_transformed.index
                                        all_df = pd.concat([all_df, 
                                                    pd.DataFrame({'Group':a.split('/')[-1].split('_')[0],
                                    'JSD':jensenshannon(tt.loc[test_set_inds].T, 
                                                        samples.loc[test_set_inds].T)
                                                        }, 
                                                       index=samples.loc[test_set_inds].index)]).ffill()

                                    if first_run:
                                        all_df.to_csv(out_path)
                                        first_run=False
                                    else:
                                        pd.concat([pd.read_csv(out_path, index_col=0), 
                                                   all_df
                                                  ]).to_csv(out_path)
if __name__=='__main__':
    main()

