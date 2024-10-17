

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
import subprocess

def set_up_prediction_eval_run(samples_,
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

    ## simulated phenotypes / covariates
    phenotypes = samples @ true_weights[:, np.newaxis]
    if phenotype_std > 0:
        phenotypes += np.random.normal(scale = phenotype_std,
                                       size=phenotypes.shape)
    
    covar_weights = np.random.normal(scale=linear_weights_stdv, 
                                     size = n_features
                                     )
    covariates = samples @ true_weights[:, np.newaxis]
    if phenotype_std > 0:
        covariates += np.random.normal(scale = phenotype_std,
                                       size=phenotypes.shape)
    
    
    studies = pd.Series( [ i // n_per_study for i in range(samples.shape[0]) 
                                     ] , 
                        index=samples.index, 
                        name='studies')

    samples_transformed = pd.DataFrame( ( rescale( samples.values *\
                                    np.power(2, true_bcfs)[studies] 
                                    )*read_depth
                              ).astype(int), 
                                       index=samples.index, 
                                       columns=samples.columns
                                      )
    
    y=phenotypes.values[:, 0] > phenotypes.median().values[0] 
    cov=covariates.values[:, 0] > covariates.median().values[0]
    
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
    
    return(samples_transformed, 
           y, 
           cov,
           studies, 
           samples
           )


def main(out_path='prediction-results-final-v3.csv'):
    
    if os.path.exists(out_path):
        raise(ValueError('out path file already exists!'))
    
    pths = ['tmp-dataset-with-added-bias/percnorm_out.csv',
            'tmp-dataset-with-added-bias/combat-out.csv',
            'tmp-dataset-with-added-bias/conqur-out.csv',
            'tmp-dataset-with-added-bias/MMUPHin_out.csv',
            'tmp-dataset-with-added-bias/PLSDAbatch_out.csv',
            'tmp-dataset-with-added-bias/voom-snm-out.csv'
           ]
    first_run=True
    
    
    for seed in range(1,26):
        for n_features in [100]:
            for n_studies in [4]:
                for n_per_study in [96]:
                    for phenotype_std in [.1]:
                        for read_depth in [1e4]:
                            data=pd.read_csv('../Simulations/generated-100-feat-datasets/dataset-seed-{}.csv'.format(seed), index_col=0)
                            
                            all_rocs=[]
                            all_names=[]
                            np.random.seed(seed)
                            torch.manual_seed(seed)
                            print(seed)
                            samples_transformed,  \
                                       y,  \
                                       cov, \
                                       studies, \
                                    samples_true \
                                = set_up_prediction_eval_run(data,
                                                 n_studies, 
                                                 n_per_study,
                                                 n_features,
                                                 phenotype_std,
                                                 read_depth, 
                                                 bcf_stdv = 2, 
                                                 linear_weights_stdv = 2,
                                                 seed=seed
                                                 )
                            
                            
                            pd.DataFrame({'label':cov, 
                                         'Study':studies}, 
                                         index=samples_transformed.index
                                        ).to_csv('tmp-dataset-with-added-bias/md.csv')
                            samples_transformed.to_csv('tmp-dataset-with-added-bias/data.csv')

                            s_relab = pd.DataFrame( 
                                           rescale( samples_transformed.values ), 
                                           index=samples_transformed.index, 
                                           columns=samples_transformed.columns )
                            raw_auroc=cross_val_score(LogisticRegression(penalty=None, 
                                                                         solver='newton-cg', 
                                                                         max_iter=1000), 
                                            X=s_relab, 
                                            y=y, 
                                            groups=studies,
                                            cv=LeaveOneGroupOut(),
                                            scoring='roc_auc',
                                            )

                            all_rocs.append(raw_auroc)
                            all_names.append(['Baseline']*n_studies)
                            
                            
                            s_relab_true = pd.DataFrame( 
                                           rescale( samples_true.values ), 
                                           index=samples_true.index, 
                                           columns=samples_true.columns )
                            true_auroc=cross_val_score(LogisticRegression(penalty=None,
                                                                          solver='newton-cg', 
                                                                          max_iter=1000 
                                                                          ), 
                                            X=s_relab_true, 
                                            y=y, 
                                            groups=studies,
                                            cv=LeaveOneGroupOut(),
                                            scoring='roc_auc', 
                                           )

                            all_rocs.append(true_auroc)
                            all_names.append(['True samples']*n_studies)

                            samples_transformed_with_batch = pd.concat([studies, 
                                                samples_transformed
                                               ], 
                                               axis=1 ).values

                            dmco=cross_val_predict(DebiasMClassifier(x_val=samples_transformed_with_batch), 
                                            X=samples_transformed_with_batch, 
                                            y=y, 
                                            groups=studies,
                                            cv=LeaveOneGroupOut(),
                                            method='predict_proba',
                                           )

                            logo=LeaveOneGroupOut()
                            all_rocs.append( [ roc_auc_score(y[test_inds],
                                                             dmco[:, 1][test_inds] )
                                              for train_inds, test_inds in 
                                             logo.split(samples_transformed_with_batch, 
                                                        y, 
                                                        studies) ] )

                            all_names.append(['DEBIAS-M']*n_studies)

                            ### run the R script
                            os.system('Rscript run-prediction-simulation-R-BC-methods.R')

                            for a in pths:
                                tt=pd.read_csv(a, index_col=0) 
                                if a not in pths[-2:]:
                                    tt=rescale(tt.values)
                                else:
                                    tt=tt.values
                                print(a)
                                sc = cross_val_score(LogisticRegression(penalty=None, 
                                                                        solver='newton-cg', 
                                                                        max_iter=1000
                                                                        ), 
                                                        X=tt, 
                                                        y=y, 
                                                        groups=studies,
                                                        cv=LeaveOneGroupOut(),
                                                        scoring='roc_auc',
                                                       )

                                all_rocs.append(sc)
                                all_names.append([a]*n_studies)


                            all_df= pd.DataFrame({'Group':all_names,
                                           'auROC':all_rocs,
                                           'N_studies':n_studies, 
                                           'N_per_study':n_per_study, 
                                           'N_features':n_features, 
                                           'Pheno_noise':phenotype_std, 
                                           'Read_depth':int(read_depth), 
                                           'run_num':seed}).explode(['Group', 'auROC'])

                            if first_run:
                                    all_df.to_csv(out_path)
                                    first_run=False
                            else:
                                pd.concat([pd.read_csv(out_path, index_col=0), 
                                           all_df]).to_csv(out_path)
if __name__=='__main__':
    main()


