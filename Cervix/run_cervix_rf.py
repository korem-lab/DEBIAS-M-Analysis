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


def flatten(l):
    return [item for sublist in l for item in sublist]

def carcinoma_func(md):
    md['label']=md['carcinoma'].fillna(False).astype(bool)
    return(md)

def CIN_func(md):
    md['label']=md['CIN']!='normal'
    return(md)

task_mapping_dict = {
                    'Cervix-carcinoma': carcinoma_func,
                    'Cervix-CIN':CIN_func
                    }

def batch_weight_feature_and_nbatchpairs_scaling(strength, df_with_batch):
    nbs = df_with_batch.iloc[:, 0].nunique()
    w = nbs * (nbs - 1) / 2
    return(strength /( w * ( df_with_batch.shape[1] - 1 ) ) )


def load_data(task):
    
    df = pd.read_csv('../data/Cervix/data.csv', index_col=0)
    md = pd.read_csv('../data/Cervix/metadata.csv', index_col=0).loc[df.index]
    md['Study'] = md.study
    md['Covariates'] = md.Age_Range


    md=task_mapping_dict[task](md)
    
    
    df=df.loc[md.index]

    df_with_batch = pd.concat([pd.Series( pd.Categorical(md.Study).codes, 
                                         index=md.index), df], axis=1)    
    y = md.label
    
    return(df, md)


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
        "clf__n_estimators": [1000],
        "clf__max_depth": [5,10,20
                          ],
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
    
    dmc = DebiasMClassifier(batch_str = actual_str, 
                            x_val=X_val.values
                            )
    dmc.fit(X_train.values, y_train)
    
    
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



name_model_map = {
                  'Random Forest Tuned': rf_tuning_approach, 
                  'DEBIAS-M into RF Tuned':debiasm_into_rf_tuning_approach,
                   }




def execute_cervix_rf_run(task, seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    df_, md = load_data(task)
    
    all_runs=[]

    for model_func_name in name_model_map:
        model_func = name_model_map[model_func_name]



        df_with_batch = pd.concat([pd.Series( pd.Categorical( md['Study'] ).codes, 
                                          index=md.index ),
                                df_.loc[md.index]], axis=1)

        y=md[['label', 'Study']]

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

            if y_val.label.nunique() == 2 :
                preds = model_func(X_train, 
                                   y_train.label.values, 
                                   X_val, 
                                   actual_str=actual_str,
                                   df_with_batch=df_with_batch
                                  )

                run_desc = {'test_batch':y_val.Study.values[0],
                            'test_batch_n':y_val.shape[0],
                            'model': model_func_name,
                            'True':y_val.label.values,
                            'Pred':preds
                           }

                all_runs.append(run_desc)
    
    return(all_runs)














