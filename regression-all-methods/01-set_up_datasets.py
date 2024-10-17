
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
import torch
import os


def set_up_datasets():
    
#     pths = ['tmp-datasets-for-R/percnorm_out.csv',
#             'tmp-datasets-for-R/combat-out.csv',
#             'tmp-datasets-for-R/conqur-out.csv',
#             'tmp-datasets-for-R/MMUPHin_out.csv',
#             'tmp-datasets-for-R/PLSDAbatch_out.csv',
#             'tmp-datasets-for-R/voom-snm-out.csv'
#            ]

    df, md = data_loading.load_Metabolites('../data/Metabolites/')
    
    pd.DataFrame({'label':md.Covariate.values, 
                  'Study':md.Study.values}, 
                  index=md.index
                 ).to_csv('tmp-datasets-for-R/md.csv')
    df.loc[md.index].to_csv('tmp-datasets-for-R/data.csv')

    ### run the R script
#     os.system('Rscript run-prediction-simulation-R-BC-methods.R')

if __name__=='__main__':
    set_up_datasets()
        
        