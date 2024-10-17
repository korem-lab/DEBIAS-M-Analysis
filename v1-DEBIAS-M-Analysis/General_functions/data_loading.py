## contains functions to load data specifc to any of our studies
## the output of each function includes the data within a consistent format

import numpy as np
import pandas as pd
import glob
import os


def load_CRC(data_path='../data/CRC/'):
    df = pd.read_csv(os.path.join(data_path, 'data_relabund.csv'), index_col=0)#.T
    md = pd.read_csv(os.path.join(data_path, 'metadata_relabund.csv'), 
                     index_col=0
                    ).loc[df.index] #.set_index('sample_id')
    md.columns=['Study', 'label', 'gender']
    md=md.loc[md.label.isin(['CRC', 'control'])]
    md.label=md.label=='CRC'
    df=df.loc[md.index]
    df = df.loc[:, ( df > 0 ).sum(axis=0)>(.05*df.shape[0]) ]
    
    md = md.loc[md.Study != "YachidaS_2019" ] 
            ## this CRC study was specific to colonoscopies, 
                ## removing as this is a very specific inclusion criteria
    
    return((df.loc[md.index], md))

def load_HIVRC(data_path):
    
    df = pd.read_csv(os.path.join(data_path,
                                   'insight.merged_otus.txt'),
                     sep='\t', 
                     index_col=0
                    ).iloc[:, :-1].T
    df = df.loc[:,((df>0).sum(axis=0) > df.shape[0]*.05 )]
    md = pd.read_csv(os.path.join(data_path,
                                  'metadata.tsv'),
                     sep='\t')
    md = md.set_index('SeqID').loc[df.index]
    
#     md['Covariate'] = md['gender']
    md['Covariate'] = ( md['Age'] > md['Age'].median() ).fillna(True)
    md['label'] = md['hivstatus']==1
    
    return((df, md))


def load_Metabolites(data_path):
    
    md = pd.read_excel(os.path.join(data_path, 'Supplementary Tables.xlsx'),#'../data/Metabolites/Supplementary Tables.xlsx', 
             sheet_name='Table S1', 
             engine='openpyxl', 
                  index_col=0)
    
    mds = []
    for fp in glob.glob(os.path.join(data_path, 'new/HSRR*.txt')):# '../data/Metabolites/new/HSRR*.txt'):
        tmp = pd.read_csv(fp, sep='\t', index_col=0)
        tmp['Study'] = fp.split('/')[-1][:-4]
        mds.append(tmp[['Study']])
    
    md = md.merge(pd.concat(mds), 
                  left_index=True, 
                  right_index=True
                  )

    df = pd.read_csv(os.path.join(data_path, 'new/otu.csv'),
#                                   '../data/Metabolites/new/otu.csv', 
                     index_col=0 ).loc[md.index]

    df = df.loc[:, ( df > 0 ).sum(axis=0)>(.05*df.shape[0]) ]

    tbl = pd.read_excel(os.path.join(data_path, 'new/41467_2019_9285_MOESM5_ESM.xlsx'),
#                                       '../data/Metabolites/new/41467_2019_9285_MOESM5_ESM.xlsx', 
                 sheet_name = 'Taxa Sequence Counts', 
                 engine='openpyxl', 
                 skiprows=3, 
                 index_col=0)[['race', 'sPTB', 'raceChar']].loc[df.index]

    tbl['label'] = tbl['sPTB']
    tbl['Covariate'] = tbl['raceChar']
    
    md = pd.concat([md, tbl], axis=1).fillna(0)
    md = md.loc[md.Study.isin(md.Study.value_counts().head(2).index)]
    
    return((df.loc[md.index], md))



## a dictionary to get the right loading function for each analysis, based only on the name
name_function_map = {a:eval('load_{}'.format(a)) for a in 
                            ['HIVRC',  
                             'Metabolites',
                             'CRC'
                            ] }













