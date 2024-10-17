

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
from sklearn.preprocessing import StandardScaler
from debiasm.torch_functions import rescale
import torch
import os


def main(out_path='res-crc-depth.csv'):
    
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

    df__, md = data_loading.load_CRC('../data/CRC/')
    md['Covariate']=md['gender']
    
    n_reps=3
    i=0 ## just a counter for seeds across runs
    depths=[1e3, 1e4, 1e5]
    
    
    study_counts=md.Study.value_counts()
    for __ in range(n_reps):
        for dd in depths:
            i+=1
            
            all_rocs=[]
            all_names=[]
            
            np.random.seed(i)
            torch.manual_seed(i)
            
            ## resampling to the specified depth
            df_tmp = pd.DataFrame( 
                        np.vstack([ np.random.multinomial(dd, a)
                                    for a in rescale(df__.values) ] ), 
                                    index=df__.index
                                  )
            
            df=pd.DataFrame(df_tmp, 
                             index=df_tmp.index)
            md_tmp=md.copy()
            
            ## presence filtering, not having that --> errors in some of the R methods
            df=df.loc[:, (df>0).max(axis=0)>0]

            df_tmp=pd.DataFrame(rescale( df.loc[md_tmp.index].values ), 
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
                
                ss=StandardScaler()
                tt=ss.fit_transform(tt)
                print(a)
                sc = cross_val_score(LogisticRegression(penalty=None), 
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

                np.random.seed(i)
                torch.manual_seed(i)
        
                ss=StandardScaler()
                df_tmp=pd.DataFrame(ss.fit_transform(df_tmp), 
                                        index=df_tmp.index, 
                                        columns=df_tmp.columns 
                                        )

                lr=LogisticRegression(penalty=None)

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


                dmc=DebiasMClassifier(x_val=df_tmp_with_batch.loc[test_inds].values)
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
                                  'Depth':dd,
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
        
        