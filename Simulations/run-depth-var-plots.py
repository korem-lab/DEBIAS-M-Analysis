

import numpy as np
import pandas as pd
from debiasm.sklearn_functions import AdaptationDebiasMClassifier, DebiasMClassifier
from debiasm.torch_functions import rescale
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
import torch
import sys
sys.path.append('../v1-DEBIAS-M-Analysis/General_functions/')
import plotting



def experiment_run(samples_,
                   n_studies, 
                   n_per_study,
                   n_features,
                   phenotype_std,
                   read_depth, 
                   bcf_stdv = 2, 
                   linear_weights_stdv = 2,
                   rd_mean=None,
                   seed=0):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
    # drop samples/features not needed for this run
    samples=samples_.sample(n=n_studies * n_per_study, 
                           replace=False)\
                .T.sample(n=n_features, replace=False).T
    
    ## simulated bcfs
    true_bcfs = np.random.normal(scale=bcf_stdv,
                                 size = (n_studies, n_features)
                                 )
    
    ## simulated linear weights
    true_weights = np.random.normal(scale=linear_weights_stdv, 
                                    size = n_features
                                    )

    ## simulated phenotypes
    phenotypes = samples @ true_weights[:, np.newaxis]
    if phenotype_std > 0:
        phenotypes += np.random.normal(scale = phenotype_std,
                                       size=phenotypes.shape)
    
    studies = pd.Series( [ i // n_per_study for i in range(samples.shape[0]) 
                                     ] , 
                        index=samples.index, 
                        name='studies')

    samples_transformed = ( rescale( samples *\
                                    np.power(2, true_bcfs)[studies] 
                                    )*read_depth
                              ).astype(int)

    test_set_inds = studies==studies.max()
    train_set_inds = studies!=studies.max()
    
    phenotypes = phenotypes.loc[train_set_inds]
    df_with_batch = pd.concat([studies, samples_transformed], axis=1 )
    
    dm = DebiasMClassifier(x_val = df_with_batch.loc[test_set_inds].values)
    dm.fit(X=df_with_batch.loc[train_set_inds].values, 
           y=phenotypes.values[:, 0] > phenotypes.median().values[0]    
          )
    
    transformed_jsds = jensenshannon(dm.transform(pd.concat([studies,
                                      samples_transformed], 
                                     axis=1 )
                          ).T,
              samples.T)

    baseline_jsds = jensenshannon(samples_transformed.T, samples.T)
    n_samples = samples.shape[0]

    if type( read_depth ) == np.ndarray:
        read_depth=rd_mean
        
    if type(bcf_stdv)==np.ndarray:
        bcf_stdv='varied'
        
    
    
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
                                   'run_num':seed}.items()]
                                )
    return(summary_df, run_description, true_bcfs, dm)







all_true_bcfs = []
all_fitted_bcfs = []
all_results = []

for n_features in [100]:
    for n_studies in [4]:
        for n_per_study in [96]:
            for phenotype_std in [0.1]:
                for read_depth in [1e3, 1e4, 1e5]:
                    for seed in range(1,26):
                        print(seed)
                        np.random.seed(seed)
                        data = pd.read_csv('../Simulations/generated-100-feat-datasets/dataset-seed-{}.csv'.format(seed), 
                                          index_col=0)
                        
                        results, run_desc, true_bcfs, dm = \
                                            experiment_run(data,
                                                           n_studies, 
                                                           n_per_study,
                                                           n_features,
                                                           phenotype_std,
                                                           read_depth, 
                                                           bcf_stdv = 2, 
                                                           seed=seed
                                                           )
            
                        all_true_bcfs.append( np.power(2,
                                             true_bcfs
                                            ) )
                        all_fitted_bcfs.append(1./np.power(2, 
                                               dm.model.batch_weights.detach().numpy()
                                              ))
                
                        all_results.append(results)














