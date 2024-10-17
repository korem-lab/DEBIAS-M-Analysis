

import numpy as np
import pandas as pd
from debiasm import DebiasMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import sys
sys.path.append('../v1-DEBIAS-M-Analysis/')
from General_functions import data_loading, plotting
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from debiasm.torch_functions import rescale
from debiasm.sklearn_functions import batch_weight_feature_and_nbatchpairs_scaling
import torch
import os

def main(out_path='res-crc-n-studies.csv'):
    
    if os.path.exists(out_path):
        raise(ValueError('out path file already exists!'))
    
    pths = ['tmp-datasets-for-R/percnorm_out.csv',
            'tmp-datasets-for-R/combat-out.csv',
            'tmp-datasets-for-R/conqur-out.csv',
            'tmp-datasets-for-R/MMUPHin_out.csv',
            'tmp-datasets-for-R/PLSDAbatch_out.csv',
            'tmp-datasets-for-R/voom-snm-out.csv'
           ]
    first_run=True

    df_, md = data_loading.load_CRC('../data/CRC/')
    md['Covariate']=md['gender']
    
    n_reps=6
    i=0 ## just a counter for seeds across runs
    study_counts=md.Study.value_counts()
    
    for __ in range(n_reps):
        i+=1
          
        for n_studies in [3, 6, 9]: ## out of 10 CRC studies


            all_rocs=[]
            all_names=[]

            np.random.seed(i)
            torch.manual_seed(i)


            # only studies w/ > 50 samples, to be consistent across all runs
            md_tmp=md.loc[md.Study.isin( 
                        study_counts.sample(n=n_studies).index
                        )]

            df=df_.loc[md_tmp.index]

            ## presence filtering, not having that --> errors in some of the R methods
            df=df.loc[:, (df>0).max(axis=0)>0]

            df_tmp=pd.DataFrame(rescale( 1+df.loc[md_tmp.index].values ), 
                                index=md_tmp.index )

            pd.DataFrame({'label':md_tmp.Covariate.values, 
                         'Study':md_tmp.Study.values}, 
                         index=md_tmp.index
                        ).to_csv('tmp-datasets-for-R/md.csv')
            df.loc[md_tmp.index].to_csv('tmp-datasets-for-R/data.csv')

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
                                                        max_iter=2500
                                                       ), 
                                     X=tt, 
                                     y=md_tmp.label, 
                                     groups=md_tmp.Study,
                                     cv=LeaveOneGroupOut(),
                                     scoring='roc_auc'
                                     )

                all_rocs += list(sc)
                all_names += [a.split('/')[-1].split('-')[0].split('_')[0]]*len(sc)

            logo=LeaveOneGroupOut()

            for train_inds, test_inds in\
                    logo.split(df_tmp, md_tmp.label, md_tmp.Study):

                train_inds=df_tmp.index[train_inds]
                test_inds=df_tmp.index[test_inds]

                if ( md_tmp.label.loc[test_inds].nunique()>1 )\
                    and ( md_tmp.label.loc[train_inds].nunique()>1):

                    np.random.seed(i)
                    torch.manual_seed(i)

                    lr=LogisticRegression(penalty=None, 
                                          max_iter=2500
                                         )

                    lr.fit(df_tmp.loc[train_inds], 
                           md_tmp.label.loc[train_inds])

                    preds=lr.predict_proba(df_tmp.loc[test_inds])


                    all_rocs.append( 
                            roc_auc_score(md_tmp.label.loc[test_inds], 
                                          preds[:, 1]
                                            ) )
                    all_names.append('Linear')

                    df_tmp_with_batch = pd.concat([
                            pd.Series( pd.Categorical(md_tmp.Study).codes, 
                                                 index=md_tmp.index), 
                                        df.loc[md_tmp.index]], 
                                                axis=1)


                    dmc=DebiasMClassifier(x_val=df_tmp_with_batch.loc[test_inds].values, 
                                         batch_str=batch_weight_feature_and_nbatchpairs_scaling(1e4, # default str settings
                                                                                                df_tmp_with_batch))
                    dmc.fit(df_tmp_with_batch.loc[train_inds].values, 
                            md_tmp.label.loc[train_inds])

                    dm_preds=dmc.predict_proba(df_tmp_with_batch.loc[test_inds].values)
                    all_rocs.append( 
                            roc_auc_score(md_tmp.label.loc[test_inds], 
                                          dm_preds[:, 1]
                                            ) )
                    all_names.append('DEBIAS-M')


            results_df_n_studies = \
                    pd.DataFrame({'auROC':all_rocs, 
                                  'Group':all_names,
                                  'N studies':n_studies,
                                  'Run num':i
                                 })

            if first_run:
                results_df_n_studies.to_csv(out_path)
                first_run=False
            else:
                pd.concat([pd.read_csv(out_path, index_col=0), 
                           results_df_n_studies]).to_csv(out_path)


    print('Successful run completed!')
    
    
    
if __name__=='__main__':
    main()
        
        
                
                
                
                
                
                