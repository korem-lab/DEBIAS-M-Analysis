

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from . import batch_weight_consistency


def HIVRC_functions(md, df_with_batch, conqur, combat):
    
    w1, w2 = batch_weight_consistency.run_batch_consistency_analysis(df_with_batch, md, seed=0)
    
    plt.figure(figsize=(12,7))
    sns.scatterplot(np.hstack( w1 ),
                    np.hstack( w2 ),
                    s=100, 
                    alpha=.1, 
                    color='black'

                   )
    plt.title( 'Log of learned weights'
                      '\n$R^2$: {:.2f}'.format(pearsonr(np.hstack( w1 ), 
                                                        np.hstack( w2 ))[0]))
    plt.xlabel('Bias Weight, All batches part 1')
    plt.ylabel('Bias Weight,  All batches part 2')
    plt.savefig('../results/HIVRC/batch-weight-consistency.pdf', 
               dpi=900, 
               format='pdf')
    
    pd.DataFrame({'w1':np.hstack( w1 ), 
                  'w2':np.hstack( w2 ), 
                  'run': np.hstack( [ np.array( [i]*w1[i].shape[0] )
                                     for i in range(len(w1)) ] )
                  }).to_csv('../results/HIVRC/batch-weight-consistency.csv')
    
    return(None)