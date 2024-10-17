

import numpy as np
import pandas as pd
from debiasm.sklearn_functions import AdaptationDebiasMClassifier
from debiasm.torch_functions import rescale
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
import torch
import os

def experiment_run(samples_,
                   n_studies, 
                   n_per_study,
                   n_features,
                   phenotype_std,
                   read_depth, 
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
    phenotypes = samples @ true_weights[:, np.newaxis]
    if phenotype_std > 0:
        phenotypes += np.random.normal(scale = phenotype_std,
                                       size=phenotypes.shape)
    
    studies = pd.Series( [ i // n_per_study for i in range(samples.shape[0]) 
                                     ] , 
                        index=samples.index, 
                        name='studies')

    samples_transformed = ( rescale(samples *\
                                    np.power(2, true_bcfs)[studies] 
                                    )*read_depth
                              ).astype(int)
    
    dm = AdaptationDebiasMClassifier()
    dm.fit(X=pd.concat([studies, samples_transformed], axis=1 ).values, 
           y=phenotypes.values[:, 0] > phenotypes.median().values[0]    
          )
    
    transformed_jsds = jensenshannon(dm.transform(pd.concat([studies,
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
    return(summary_df, run_description)



def main():
    for n_features in [1000, 100, 10]:#, 1000]:
        for n_studies in [4,3,2]:
            for n_per_study in [96, 72, 48, 24]:
                for read_depth in [1e4, 1e3,  1e5]:
                    for phenotype_std in [0, .1, 1, 10]:
                        for seed in range(1,26):
                            data = pd.read_csv('generated_datasets/dataset-seed-{}.csv'.format(seed), 
                                              index_col=0)
                            try:
                                results, run_desc = experiment_run(data,
                                                                   n_studies, 
                                                                   n_per_study,
                                                                   n_features,
                                                                   phenotype_std,
                                                                   read_depth, 
                                                                   bcf_stdv = 2, 
                                                                   seed=seed
                                                                   )

                                if os.path.exists('../../results/Simulations/all-simulation-runs/{}.csv'\
                                                .format(run_desc)) == False:

                                    results.to_csv('../../results/Simulations/all-simulation-runs/{}.csv'\
                                                    .format(run_desc))
                            except:
                                pass
                            
                            
                            
                            
if __name__=='__main__':
    main()