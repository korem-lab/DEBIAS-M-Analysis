import numpy as np
import pandas as pd
import torch
from debiasm import DebiasMClassifier
import time
def flatten(xss):
    return [x for xs in xss for x in xs]


def set_up_input(smps_per_batch, 
                n_batches,
                n_features):
    total_smps=smps_per_batch*n_batches
    dataset=np.random.rand(total_smps, 
                           n_features
                           )
    
    batches = np.array( flatten([[i]*smps_per_batch
                        for i in range(n_batches)
                      ]) )[:, np.newaxis]

    X_with_batch = np.hstack([batches, 
                              dataset])

    y = np.random.rand(total_smps)>0.5
    
    return(X_with_batch, y)
    
    
    
def get_runtime(X_with_batch, y, smps_per_batch):
    start=time.time()
    dmc=DebiasMClassifier(x_val=X_with_batch[-smps_per_batch:])
    dmc.fit(X_with_batch[:-smps_per_batch], 
            y[:-smps_per_batch])
    runtime=time.time()-start
    return(runtime)



def main(out_path='runtime-n-per-study-results.csv', 
         seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    nbs=[]
    runtimes=[]
    n_batches=8

    for smps_per_batch in [24, 48, 96, 256, 512]:
        for __ in range(10):
            X_with_batch, y = set_up_input(smps_per_batch, 
                                           n_batches,
                                           100
                                           )
            rt=get_runtime(X_with_batch, y, smps_per_batch)

            nbs.append(smps_per_batch)
            runtimes.append(rt)

        pd.DataFrame({'Number of batches':n_batches, 
                      'Samples per batch':nbs, 
                      'Number of taxa':100, 
                      'Runtime (s)':runtimes}
                     ).to_csv(out_path)
        
        
if __name__=='__main__':
    main()
    
    