
import numpy as np
import pandas as pd
import os
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from SBCMP import rescale
from sklearn.metrics import roc_auc_score
from debiasm import DebiasMClassifier
import debiasm
from sklearn.ensemble import RandomForestClassifier
import torch.nn.functional as F
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from main import batch_weight_feature_and_nbatchpairs_scaling



def load_Segata_PFS12(data_path = '../data/Segata/'):
    md = pd.read_csv( os.path.join(data_path,'metadata.csv') ).set_index('sample_id')
    md['study_name'] = md['subcohort']
    md.loc[md.subcohort=='Leeds', 'study_name'] = md.loc[md.subcohort=='Leeds', 'location']

    
    md_other = pd.read_csv( os.path.join(data_path,'metadata_other.csv') ).set_index('sample_id')
    
    md = md.loc[md.PFS12.isna()==False]
    md=md.dropna(axis=1)
    df = pd.read_csv( os.path.join(data_path,'data.csv'), index_col=0).loc[md.index]
    df_other = pd.read_csv( os.path.join(data_path,'data_other.csv'), 
                           index_col=0).loc[md_other.index]
    md=pd.concat([md, md_other], axis=0)
    
    md = md.loc[md.PFS12.isna()==False]
    
    md=md.loc[md.study_name.isin([
                                  'PRIMM-NL', 
                                  'PRIMM-UK', 
                                  'Leeds', 
                                  'Manchester',
                                  'WindTT_2020', 
                                  'PetersBA_2019']
                                  )]
    
   
    df=pd.concat([df, df_other], axis=0).fillna(0)
    
    df=df.loc[md.index]
                     
    df_ = df.copy()
    df_ = df_.loc[:, ( (df_  > 0 ).sum(axis=0) > df.shape[0]*0.05 ) ]
    
    ## filter added in original analysis
    df_ = df_.loc[:, ( rescale( df_ ).max(axis=0) >= 1e-4 ) ] 
    
    md['label'] = md['PFS12'] != 'yes'
    
    return( (rescale(df_), md) )



def rf_tuning_approach(X_train, 
                       y_train, 
                       X_val,  
                       actual_str=None,
                       df_with_batch=None, 
                       **kwargs
                       ):
    
    
    pipe = Pipeline([
                ("clf", RandomForestClassifier())
                ])
    
    # Declare a hyperparameter grid
    param_grid = {
        "clf__n_estimators": [500],
        "clf__max_depth": [5,10, 20],
        'clf__max_features': [ .5, .75, 1],
    }

    rf = GridSearchCV(pipe,
                      param_grid=param_grid,
                      cv=3,
                      n_jobs=1, 
                      verbose=False,
                      scoring='roc_auc')
    
    ss=StandardScaler()
    rf.fit( ss.fit_transform( rescale(X_train.iloc[:, 1:].values) ), y_train)
    preds = rf.predict_proba( ss.transform( rescale(X_val.iloc[:, 1:].values)) )
    return(preds[:, 1])




def debiasm_into_rf_tuning_approach(X_train, 
                                    y_train, 
                                    X_val, 
                                    actual_str=None,
                                    df_with_batch=None, 
                                    **kwargs
                                    ):
    
    
    pipe = Pipeline([
                        ("DebiasM", DebiasMClassifier())
                    ])

    # Declare a hyperparameter grid
    param_grid = {
               'DebiasM__x_val':[np.vstack((X_train.values, 
                                           X_val.values))],
               'DebiasM__learning_rate':[.0005, .005],
               'DebiasM__l2_strength':[0, 1e-7,  1e-5, 1e-3], 
               'DebiasM__w_l2':[0, 1e-7, 1e-5, 1e-3],
               'DebiasM__batch_str':[actual_str, actual_str/10, actual_str*10],
        }

    # Perform grid search, fit it, and print score
    gs = GridSearchCV(pipe, 
                      param_grid=param_grid,
                      n_jobs=1,
                      verbose=False,
                      scoring='roc_auc'
                     )

    gs.fit(X_train.values, y_train)

    dmc = gs.best_estimator_[0]
    
    
    x=torch.tensor(df_with_batch.values).float()
    batch_inds, x = x[:, 0], x[:, 1:]

    x = F.normalize( torch.pow(2, dmc.model.batch_weights[batch_inds.long()] ) * x, p=1 )
    df_wb_tmp=df_with_batch.copy()
    df_wb_tmp.iloc[:,1:]=x.detach().numpy()
    train_inds = X_train.index
    val_inds=X_val.index
    X_train__, X_val__ = df_wb_tmp.loc[train_inds], \
                                df_wb_tmp.loc[val_inds]

                
    return( rf_tuning_approach(X_train__, y_train, X_val__) )

def linear_clr_training_approach(X_train, y_train, X_val,  
                                    actual_str=None,
                                    df_with_batch=None, 
                                    **kwargs):
    
    cl_ts = np.log10( 1e-4 + np.vstack((X_train.values, X_val.values))[:, 1:] )
    
    cl_ts = np.hstack((np.hstack((X_train.values[:, 0],
                                  X_val.values[:, 0]
                                  ))[:, np.newaxis], 
                       cl_ts))
    
    
    X_train = cl_ts[:X_train.shape[0]]
    X_val = cl_ts[X_train.shape[0]:]
    
    ss=StandardScaler()
    rf = LogisticRegressionCV(penalty = 'l1', 
                              Cs =np.logspace(-3, 3, 10),
                              solver = 'liblinear', 
                              cv=5,
                              )
    
    
    rf.fit( ss.fit_transform( X_train ), y_train)
    preds = rf.predict_proba( ss.transform( X_val ) )
    
    return(preds[:, 1])

def debiasm_into_clr_lr_training_approach(X_train, 
                                      y_train, 
                                      X_val, 
                                      actual_str=None,
                                      df_with_batch=None, 
                                      **kwargs
                                      ):
    
    
    pipe = Pipeline([
                        ("DebiasM", DebiasMClassifier())
                    ])

    # Declare a hyperparameter grid
    param_grid = {
               'DebiasM__x_val':[np.vstack((X_train.values, 
                                           X_val.values))],
               'DebiasM__learning_rate':[.0005, .005],
               'DebiasM__l2_strength':[0, 1e-7,  1e-5, 1e-3], 
               'DebiasM__w_l2':[0, 1e-7, 1e-5, 1e-3],
               'DebiasM__batch_str':[actual_str, actual_str/10, actual_str*10],
        }

    # Perform grid search, fit it, and print score
    gs = GridSearchCV(pipe, 
                      param_grid=param_grid,
                      cv=3, 
                      n_jobs=1,
                      verbose=False,
                      scoring='roc_auc'
                     )

    gs.fit(X_train.values, y_train)

    dmc = gs.best_estimator_[0]
    
    x=torch.tensor(df_with_batch.astype(float).values)
    batch_inds, x = x[:, 0], x[:, 1:]

    x = F.normalize( torch.pow(2, dmc.model.batch_weights[batch_inds.long()] ) * x, p=1 )
    df_wb_tmp=df_with_batch.copy()
    df_wb_tmp.iloc[:,1:]=x.detach().numpy()
    train_inds = X_train.index
    val_inds=X_val.index
    X_train_debias, X_val_debias = df_wb_tmp.loc[train_inds],\
                                        df_wb_tmp.loc[val_inds]

                
    return( linear_clr_training_approach(X_train_debias, 
                                       y_train,
                                       X_val_debias) 
          )




name_model_map = {
                  'DEBIAS-M into Log-linear':debiasm_into_clr_lr_training_approach,
                  'Random Forest Tuned': rf_tuning_approach, 
                  'DEBIAS-M into RF Tuned':debiasm_into_rf_tuning_approach,
                   }


name_model_map = {
                  'Log-Linear':linear_clr_training_approach,
                 'DEBIAS-M into  Log-linear':debiasm_into_clr_lr_training_approach,
                  'Random Forest': rf_tuning_approach, 
                  'DEBIAS-M into RF':debiasm_into_rf_tuning_approach,
                   }



def main():
    df_, md = load_Segata_PFS12()
    df_=df_.loc[md.index]

    md.label = md.label.astype(int)
    all_runs=[]
    seed=1

    for model_func_name in name_model_map:
        model_func = name_model_map[model_func_name]



        df_with_batch = pd.concat([pd.Series( pd.Categorical( md['study_name'] ).codes, 
                                          index=md.index ),
                                df_.loc[md.index]], axis=1)

        y=md[['label', 'study_name']]

        inds = y['label'] != -1
        y=y.loc[inds]
        df_with_batch = df_with_batch.loc[inds]

        actual_str = batch_weight_feature_and_nbatchpairs_scaling(1e4, 
                                                                  df_with_batch)

        for batch in df_with_batch.iloc[:, 0].unique():
            np.random.seed(seed)
            torch.manual_seed(seed)
            val_inds = df_with_batch.iloc[:, 0]==batch
            X_train, X_val = df_with_batch[~val_inds], df_with_batch[val_inds]
            y_train, y_val = y[~val_inds], y[val_inds]

            preds = model_func(X_train, 
                               y_train.label.values, 
                               X_val, 
                               actual_str=actual_str,
                               df_with_batch=df_with_batch
                              )

            run_desc = {'test_batch':y_val.study_name.values[0],
                        'test_batch_n':y_val.shape[0],
                        'model': model_func_name,
                        'auroc':roc_auc_score(y_val.iloc[:, 0].values, 
                                              preds)
                       }

            all_runs.append(run_desc)



    pd.set_option('display.precision', 2)
    order = ['PRIMM-NL', 'PRIMM-UK', 
             'Leeds', 'Manchester', 'WindTT_2020', 'PetersBA_2019', 'Barcelona']


    tmp_df_ = pd.DataFrame.from_dict(  all_runs )\
                        .pivot(index='model', 
                               columns='test_batch', 
                               values='auroc').loc[
                         name_model_map.keys()]


    qq = tmp_df_.round(2)\
            .style.background_gradient(cmap ='rocket', axis=None, vmin=.5, vmax=1)\
            .set_properties(**{'font-size': '20px'})

    qq.to_excel('../results/Segata/finalized-analysis.xlsx')

    
    
if __name__=='__main__':
    main()






