import numpy as np
import pandas as pd
import torch
import SBCMP

def batch_weight_feature_and_nbatchpairs_scaling(strength, df_with_batch):
    nbs = df_with_batch.iloc[:, 0].nunique()
    w = nbs * (nbs - 1) / 2
    return(strength /( w * ( df_with_batch.shape[1] - 1 ) ) )




def test_heldout_batch_weight(val_batch, 
                              df_with_batch, 
                              md, 
                              seed=0
                             ):
    np.random.seed(seed)
    val_part1 = df_with_batch.loc[df_with_batch[0]==val_batch].sample(frac=.5)
    val_part2 =  df_with_batch.loc[(df_with_batch[0]==val_batch)&
                                   (df_with_batch.index.isin(val_part1.index)==False)]

    train_t1 = pd.concat([df_with_batch.loc[df_with_batch[0]!=val_batch], 
                          val_part1], axis=0)
    train_y_t1 = md.loc[train_t1.index].label

    val_y_t1 = md.loc[val_part1.index].label
    
    b_str = batch_weight_feature_and_nbatchpairs_scaling(1e4, df_with_batch)

    mm_1, mod1 = SBCMP.SBCMP_train_and_pred(train_t1.values, 
                                            val_part1.values, 
                                            train_y_t1.values, 
                                            val_y_t1.values, 
                                            batch_sim_strength=b_str,
#                                             learning_rate=.1,
                                            learning_rate=.005,
                                            test_split=0,
                                            min_epochs=15,
                                            )

    train_t2 = pd.concat([df_with_batch.loc[df_with_batch[0]!=val_batch], 
                          val_part2], axis=0)
    train_y_t2 = md.loc[train_t2.index].label

    val_y_t2 = md.loc[val_part2.index].label


    mm_2, mod2 = SBCMP.SBCMP_train_and_pred(train_t2.values, 
                                            val_part2.values, 
                                            train_y_t2.values, 
                                            val_y_t2.values, 
                                             batch_sim_strength=b_str,
#                                             learning_rate=.1,
                                            learning_rate=.005,
                                            test_split=0,
                                            min_epochs=15,
                                            )

    return(mod1, mod2)


def run_batch_consistency_analysis(df_with_batch, md, seed=0):

    mods=[]

    for vb in df_with_batch[0].unique():
        mods.append( test_heldout_batch_weight(vb, 
                                               df_with_batch, 
                                               md, 
                                               seed=seed
                                               )
                   )

    w1s = []
    w2s = []

    for i in range(len(mods)):

        w1=mods[i][0].batch_weights[df_with_batch[0].unique()[i]].detach().numpy()
        w2=mods[i][1].batch_weights[df_with_batch[0].unique()[i]].detach().numpy()

        # ignoring features that were only present in 1/2 of the batch
        mask = ( w1 != 0 ) * (w2 != 0)
        w1, w2 = w1[mask], w2[mask]

        w1s.append(w1)
        w2s.append(w2)
        
        
    return(w1s, w2s)