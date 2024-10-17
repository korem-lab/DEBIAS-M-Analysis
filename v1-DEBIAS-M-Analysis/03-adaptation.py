#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import torch

from General_functions import data_loading, plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, roc_auc_score, roc_curve
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.base import BaseEstimator
from SBCMP import rescale
import SBCMP

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import SBCMP
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import datasets
import numpy as np
from debiasm.torch_functions import pairwise_distance

from debiasm import AdaptationDebiasMClassifier


import numpy as np
import pandas as pd
from General_functions import all_classification_pipelines, plotting


def flatten(l):
    return [item for sublist in l for item in sublist]

def carcinoma_func(md):
    md['label']=md['carcinoma'].fillna(False).astype(bool)
    return(md)

def CIN_func(md):
    md['label']=md['CIN']!='normal'
    return(md)

task_mapping_dict = {
                    'Cervix-CIN':CIN_func,
                    'Cervix-carcinoma': carcinoma_func,
                    }

def batch_weight_feature_and_nbatchpairs_scaling(strength, df_with_batch):
    nbs = df_with_batch.iloc[:, 0].nunique()
    w = nbs * (nbs - 1) / 2
    return(strength /( w * ( df_with_batch.shape[1] - 1 ) ) )

def main():

    df = pd.read_csv('../data/Cervix/data.csv', index_col=0)
    md = pd.read_csv('../data/Cervix/metadata.csv', index_col=0).loc[df.index]
    md['Study'] = md.study
    md['Covariates'] = md.Age_Range


    task='Cervix-CIN'

    md['Study'] = md.study
    md['Covariates'] = md.Age_Range


    md=task_mapping_dict[task](md)
    
    df=df.loc[md.index]

    df_with_batch = pd.concat([pd.Series( pd.Categorical(md.Study).codes, 
                                         index=md.index), df], axis=1)    
    y=md.label

    train_inds = md.Study != 'Paola'
    
    seed=123
    np.random.seed(seed)
    torch.manual_seed(seed)

    dmc = AdaptationDebiasMClassifier()
    dmc = dmc.fit(df_with_batch.loc[train_inds].values, md.label.loc[train_inds].values)


    x_base = pd.concat([ df_with_batch.loc[train_inds].iloc[:, 0], 
                dmc.transform(df_with_batch.loc[train_inds])], 
              axis=1 )

    x_base_means = F.normalize( torch.tensor( x_base.groupby(0)[x_base.columns].mean()\
                                                     .iloc[:, 1:].values ), p=1 )
    x_test = torch.Tensor( df_with_batch.loc[~train_inds].iloc[:, 1:].values
                                 )
    input_dim=x_test.shape[1]
    test_bcfs = torch.nn.Parameter(data = torch.zeros(1,
                                                      input_dim)
                                  )


    optimizer=torch.optim.Adam( [test_bcfs], 
                               lr = 0.005
                               )

    losses = []
    aurocs = []
    for i in range(10000):
        xt1 = F.normalize( x_test*torch.pow(2, test_bcfs), p=1)
        test_ps = F.softmax( dmc.model.linear(xt1), dim=1 )[:, 1].detach().numpy()
        aaau = roc_auc_score( md.loc[~train_inds].label.values, 
                           test_ps 
                         )
        aurocs.append( aaau )

        if i%1000==0 and i > 0:
            print(loss)
            test_ps = F.softmax( dmc.model.linear(xt1), dim=1 )[:, 1].detach().numpy()
            print( roc_auc_score( md.loc[~train_inds].label.values, 
                           test_ps 
                         )
                 ) 


        xt1 = F.normalize( x_test*torch.pow(2, test_bcfs), p=1)
        loss = sum( [pairwise_distance(xt1, a) for a in x_base_means] ).sum()*0.01
        optimizer.zero_grad()
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        # updateing the parameters after each iteration
        optimizer.step()
        losses.append(loss.item())


    plt.figure(figsize=(8,8))
    sns.lineplot(x = list( range(len(aurocs)) ), 
                 y = losses, 
                 linewidth=5
                 )
    plt.ylabel('Test adjustment loss')
    plt.xlabel('Test adjustment iteration')
    plt.savefig('../results/Cervix-carcinoma/adaptation/adaptation-loss-with-axes.pdf', 
                format='pdf', 
                dpi=900,
               bbox_inches='tight')


    plt.figure(figsize=(8,8))
    ax = sns.lineplot(x = list( range(len(aurocs)) ), 
                 y = losses, 
                 linewidth=5
                 )
    plt.ylabel(None)
    plt.xlabel(None)
    ax.set(yticklabels=[], 
           xticklabels=[]
           )
    plt.savefig('../results/Cervix-carcinoma/adaptation/adaptation-loss-no-axes.pdf', 
                format='pdf', 
                dpi=900,
               bbox_inches='tight')



    plt.figure(figsize=(8,8))
    sns.lineplot(x = list( range(len(aurocs)) ), 
                 y = aurocs, 
                linewidth=5)
    plt.ylabel('Test auROCs')
    plt.xlabel('Test adjustment iteration')
    plt.savefig('../results/Cervix-carcinoma/adaptation/adaptation-auroc-with-axes.pdf', 
                format='pdf', 
                dpi=900,
               bbox_inches='tight')
    plt.show()


    plt.figure(figsize=(8,8))
    ax=sns.lineplot(x = list( range(len(aurocs)) ), 
                 y = aurocs, 
                linewidth=5)
    plt.ylabel(None)
    plt.xlabel(None)
    ax.set(yticklabels=[], 
           xticklabels=[]
           )

    plt.savefig('../results/Cervix-carcinoma/adaptation/adaptation-auroc-no-axes.pdf', 
                format='pdf', 
                dpi=900,
               bbox_inches='tight')
    plt.show()
    
import numpy as np
import pandas as pd
from debiasm.sklearn_functions import AdaptationDebiasMClassifier

import pandas as pd
import os
from General_functions.data_loading import name_function_map
name_function_map['CRC_with_labels'] = name_function_map['CRC']
from General_functions import all_classification_pipelines
from General_functions import plotting #import save_roc_boxplots

## the below functions include analyses specific to a particular dataset
from CRC.main import CRC_functions
from HIVRC.main import HIVRC_functions
from Cervix import run_cervix_analysis
from Metabolites.main import Metabolites_functions
CRC_with_labels_functions = CRC_functions ## both of these are just placeholders
import torch
from sklearn.metrics import roc_auc_score

def flatten(l):
    return [item for sublist in l for item in sublist]


def batch_weight_feature_and_nbatchpairs_scaling(strength, df_with_batch):
    nbs = df_with_batch.iloc[:, 0].nunique()
    w = nbs * (nbs - 1) / 2
    return(strength /( w * ( df_with_batch.shape[1] - 1 ) ) )

    
def general_adaptation_main():
    
    task_list = [
                   'CRC', 
                   'HIVRC', 
                 ]
    ## for data setup
    data_path_map = {a:'../data/{}'.format(a) for a in task_list }
    
    seed=123
    resulting_aurocs = []
    result_task_names = []

    for task in task_list:

        dataset=name_function_map[task](data_path_map[task])

        df = dataset[0]
        md = dataset[1]

        df_with_batch = pd.concat([pd.Series( pd.Categorical(md.Study).codes, 
                                             index=md.index), df], axis=1)    
        y=md.label
        inds=df.sum(axis=1) > 10

        df_with_batch = df_with_batch.loc[inds]
        y=y.loc[inds]

        np.random.seed(seed)
        torch.manual_seed(seed)
        for batch in df_with_batch[0].unique():


            val_inds = df_with_batch[0]==batch
            X_train, X_val = df_with_batch[~val_inds], df_with_batch[val_inds]
            y_train, y_val = y[~val_inds], y[val_inds]


            if (np.unique( y_val ).shape[0]>1):
                dmca = AdaptationDebiasMClassifier()
                dmca.fit(X_train.values, y_train)
                preds = dmca.predict_proba(X_val.values[:, 1:])

                resulting_aurocs.append(roc_auc_score(y_val, preds[:, 1]))
                result_task_names.append(task)
                
                
    for task in ['Cervix-CIN']:
        df = pd.read_csv('../data/Cervix/data.csv', index_col=0)
        md = pd.read_csv('../data/Cervix/metadata.csv', index_col=0).loc[df.index]
        md['Study'] = md.study
        md['Covariates'] = md.Age_Range

        md=CIN_func(md)

        df=df.loc[md.index]

        df_with_batch = pd.concat([pd.Series( pd.Categorical(md.Study).codes, 
                                             index=md.index), df], axis=1)    
        y=md.label
        inds= ( df.sum(axis=1)*100 ) > 10

        df_with_batch=df_with_batch.loc[inds].astype(float)
        y=y.loc[inds].astype(int)

        np.random.seed(seed)
        torch.manual_seed(seed)
        for batch in df_with_batch[0].unique():


            val_inds = df_with_batch[0]==batch
            X_train, X_val = df_with_batch[~val_inds], df_with_batch[val_inds]
            y_train, y_val = y[~val_inds], y[val_inds]


            if (np.unique( y_val ).shape[0]>1):
                dmca = AdaptationDebiasMClassifier()
                dmca.fit(X_train.values, y_train)
                preds = dmca.predict_proba(torch.tensor(X_val.values[:, 1:]).float() ) 

                resulting_aurocs.append(roc_auc_score(y_val, preds[:, 1]))
                result_task_names.append(task)

    
    import glob
    from General_functions import plotting
    import seaborn as sns
    import matplotlib.pyplot as plt

    dm_standard = pd.concat([pd.read_csv(a, index_col=0).query('Group=="debias-m"')\
                                                    .assign(Task=a.split('/')[-2])
                             for a in glob.glob('../results/*/auroc-boxplot-linear.csv')], 
                            axis=0
                           ).dropna(axis=1)
    dm_standard.Group = dm_standard.Group.str.upper()
    dm_standard = dm_standard.loc[dm_standard.Task.str.contains('labels')==False]

    all_results = pd.concat([dm_standard, 
                       pd.DataFrame({'Task':result_task_names, 
                          'auROC':resulting_aurocs, 
                          'Group': "Adaptation DEBIAS-M"})
                      ])


    all_results.to_csv('../results/Cervix-carcinoma/adaptation/general-adaptation-benchmark.csv')


    palette = plotting.global_palette
    palette['DEBIAS-M']=palette['debias-m']
    palette['Adaptation DEBIAS-M']='#2C60EA'

    plt.figure(figsize=(16,8))
    ax = sns.boxplot(y='auROC', 
                     x='Task', 
                     hue='Group', 
                     data=all_results, 
                     palette=palette,
                     fliersize=0)
    handles, labels = ax.get_legend_handles_labels()
    sns.swarmplot(y='auROC', 
                  x='Task', 
                  hue='Group', 
                  data=all_results, 
                  color='black', 
                  dodge=True, 
                  size=10
                  )
    ax.legend(handles[:2], labels[:2], loc='lower left')
    plt.savefig('../results/Cervix-carcinoma/adaptation/general-adaptation-benchmark.pdf', 
                dpi=900, 
                format='pdf', 
                bbox_inches='tight')

    plt.figure(figsize=(16,8))
    ax = sns.boxplot(y='auROC', 
                     x='Task', 
                     hue='Group', 
                     data=all_results, 
                     palette=palette,
                     fliersize=0)
    handles, labels = ax.get_legend_handles_labels()
    sns.swarmplot(y='auROC', 
                  x='Task', 
                  hue='Group', 
                  data=all_results, 
                  color='black', 
                  dodge=True, 
                  size=10
                  )

    ax.legend(handles[:2], labels[:2], loc='lower left')
    plt.ylabel(None)
    plt.xlabel(None)
    ax.set(yticklabels=[], 
           xticklabels=[]
           )
    plt.legend().remove()

    plt.savefig('../results/Cervix-carcinoma/adaptation/general-adaptation-benchmark-no-axes.pdf', 
                dpi=900, 
                format='pdf', 
                bbox_inches='tight')



    from scipy.stats import wilcoxon
    for tsk in all_results.Task.unique():
        print(tsk)
        print(
            wilcoxon(
                    all_results.loc[(all_results.Task==tsk)&
                                    (all_results.Group=='DEBIAS-M')
                                   ].auROC, 
                    all_results.loc[(all_results.Task==tsk)&
                                        (all_results.Group=='Adaptation DEBIAS-M')
                                       ].auROC , 
            alternative='greater'
                    ))
        
        
if __name__=='__main__':
    main()
    general_adaptation_main()
