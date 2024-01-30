
import numpy as np
import pandas as pd
import sys
from General_functions import all_classification_pipelines, plotting
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from debiasm import DebiasMClassifier
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def flatten(l):
    return [item for sublist in l for item in sublist]

eps=1e-10
def batch_weight_feature_and_nbatchpairs_scaling(strength, df_with_batch):
    nbs = df_with_batch.iloc[:, 0].nunique()
    w = nbs * (nbs - 1) / 2
    return(strength /( w * ( df_with_batch.shape[1] - 1 ) ) )

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.functional import softmax, pairwise_distance
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics.functional import accuracy

from pl_bolts.datamodules import SklearnDataModule

from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple, Type
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def rescale(xx):
    return(xx/xx.sum(axis=1)[:, np.newaxis] )

def to_categorical(y, num_classes=2):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


class PL_SBCMP_multitask(pl.LightningModule):
    """Mutitask DEBIAS-M model"""

    def __init__(
        self,
        X, 
        batch_sim_strength: float,
        input_dim: int,
        num_classes: int,
        bias: bool = True,
        learning_rate: float = 1e-4,
        optimizer: Type[Optimizer] = Adam,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
        w_l2 : float = 0.0,
        n_tasks = 2,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            input_dim: number of dimensions of the input (at least 1)
            num_classes: number of class labels (binary: 2, multi-class: >2)
            bias: specifies if a constant or intercept should be fitted (equivalent to fit_intercept in sklearn)
            learning_rate: learning_rate for the optimizer
            optimizer: the optimizer to use (default: ``Adam``)
            l1_strength: L1 regularization strength (default: ``0.0``)
            l2_strength: L2 regularization strength (default: ``0.0``)
        """
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        
        if n_tasks <=1:
            raise(ValueError('only "{}" tasks specified, use base function if only a single task is needed'.format(n_tasks)))
        
        self.linear_weights = torch.nn.ModuleList([ 
                        nn.Linear(in_features=self.hparams.input_dim, 
                                  out_features=self.hparams.num_classes, 
                                  bias=bias) 
                                  for task in range(n_tasks) ])

        self.X=X[:, 1:]
        self.bs=X[:, 0].long()
        self.unique_bs=self.bs.unique().long()
        self.n_batches=self.unique_bs.max()+1
        self.batch_weights = torch.nn.Parameter(data = torch.zeros(self.n_batches,
                                                                   input_dim))

        self.batch_sim_str=batch_sim_strength
        
        
    def forward(self, x: Tensor) -> Tensor:
        batch_inds, x = x[:, 0], x[:, 1:]
        x = F.normalize( torch.pow(2, self.batch_weights[batch_inds.long()] ) * x, 
                        p=1 )
        
        # a separate linear / softmax layer for each task
        y_hats = [softmax(self.linear_weights[i](x))
                  for i in range(self.hparams.n_tasks)]
        return y_hats

    
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch
        
        # flatten any input
        x = x.view(x.size(0), -1)

        y_hats = self.forward(x)
        
        # PyTorch cross_entropy function combines log_softmax and nll_loss in single function
        loss = sum( [ F.cross_entropy(y_hats[i][y[:, i]!=-1], 
                                      y[:, i][y[:, i]!=-1], 
                                      reduction="sum"
                                     )
                      for i in range(self.hparams.n_tasks) 
                    ] )
        
        # L1 regularizer
        if self.hparams.l1_strength > 0:
            l1_reg = sum([ linear.weight.abs().sum()
                            for linear in self.linear_weights ])
            loss += self.hparams.l1_strength * l1_reg
        
        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = sum([ linear.weight.pow(2).sum() \
                              for linear in self.linear_weights ])
            loss += self.hparams.l2_strength * l2_reg
            
        # DOMAIN similarity regularizer / bias correction
        if self.batch_sim_str > 0:
            x1 = torch.stack( [ ( torch.pow(2, self.batch_weights\
                                          )[torch.where(self.unique_bs==a)[0]] * \
                    (self.X[ torch.where(self.bs==a)[0] ] )  \
                   ).mean(axis=0) for a in self.unique_bs ] )

            x1=F.normalize(x1, p=1)

            loss += sum( [pairwise_distance(x1, a) for a in x1] ).sum() *\
                                    self.batch_sim_str


            
        # L2 regularizer for bias weight    
        if self.hparams.w_l2 > 0:
            # L2 regularizer for weighting parameter
            l2_reg = self.batch_weights.pow(2).sum()
            loss += self.hparams.w_l2 * l2_reg
            
        

        loss /= float( x.size(0) )
        return {"loss": loss}

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hats = self.forward(x)
                               
         # PyTorch cross_entropy function combines log_softmax and nll_loss in single function
        loss = sum( [ F.cross_entropy(y_hats[i][y[:, i]!=-1], 
                                      y[:, i][y[:, i]!=-1], 
                                      reduction="sum"
                                     )
                      for i in range(self.hparams.n_tasks) 
                    ] )
        
        # L1 regularizer
        if self.hparams.l1_strength > 0:
            l1_reg = sum([ linear.weight.abs().sum()
                            for linear in self.linear_weights ])
            loss += self.hparams.l1_strength * l1_reg
        
        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = sum([ linear.weight.pow(2).sum() \
                              for linear in self.linear_weights ])
            loss += self.hparams.l2_strength * l2_reg
        
        self.log('val_loss', loss)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        acc = 0
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        
        return {"val_loss": val_loss}

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)
        acc = 0#
        return {"test_loss": F.cross_entropy(y_hat, y),
                "acc": acc}

    def test_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        acc = 0
        test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        tensorboard_logs = {"test_ce_loss": test_loss, "test_acc": acc}
        progress_bar_metrics = tensorboard_logs
        return {"test_loss": test_loss,
                "log": tensorboard_logs,
                "progress_bar": progress_bar_metrics}

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        parser.add_argument("--input_dim", type=int, default=None)
        parser.add_argument("--num_classes", type=int, default=None)
        parser.add_argument("--bias", default="store_true")
        parser.add_argument("--batch_size", type=int, default=100)
        
        return parser
    
    
    
    
def SBCMP_mutlitask_train_and_pred(X_train, 
                                   X_val, 
                                   y_train, 
                                   y_val,
                                   batch_sim_strength=1, 
                                   batch_size=None,
                                   w_l2 = 0,
                                   learning_rate=0.005,
                                   l2_strength=0,
                                   includes_batches=False,
                                   val_split=0.1, 
                                   test_split=0,
                                   min_epochs=15
                                   ):
    n_tasks = y_train.shape[1]
    
    if batch_size is None:
        batch_size=X_train.shape[0]
    
    baseline_mods=[]
    for i in range(n_tasks):

        inds_tmp = y_train[:, i] > -1
        baseline_mod = LogisticRegression(max_iter=2500)
        baseline_mod.fit(rescale( X_train[:, 1:][inds_tmp]), 
                         y_train[:, i][inds_tmp].astype(int))
        baseline_mods.append(baseline_mod)
        
        
    model = PL_SBCMP_multitask(X = torch.tensor( np.vstack((X_train, X_val)) ),
                               batch_sim_strength = batch_sim_strength,
                               input_dim = X_train.shape[1]-1, 
                               num_classes = 2, 
                               batch_size = batch_size,
                               learning_rate = learning_rate,
                               l2_strength = l2_strength,
                               n_tasks=n_tasks,
                               w_l2 = w_l2
                               )
    
    # initialize parameters to lbe similar to standard logistic regression
    for i in range(n_tasks):
        try:
            model.linear_weights[i].weight.data[0]= \
                            -torch.tensor( baseline_mods[i].coef_[0] )
            model.linear_weights[i].weight.data[1]= \
                             torch.tensor( baseline_mods[i].coef_[0] )
        except:
            pass

    ## build pl dataloader
    dm = SklearnDataModule(X_train, 
                           y_train.astype(int),
                           val_split=val_split,
                           test_split=test_split
                           )

    ## run training
    trainer = pl.Trainer(callbacks=[EarlyStopping(monitor="val_loss",
                                                  mode="min", 
                                                  patience=2)], 
                         check_val_every_n_epoch=2, 
                         weights_summary=None, 
                         progress_bar_refresh_rate=1, 
                         min_epochs=min_epochs
                            )
    trainer.fit(model, 
                train_dataloaders=dm.train_dataloader(), 
                val_dataloaders=dm.val_dataloader()
               )
    try:
        ## get val predictions
        val_preds = model.forward( torch.tensor( X_val.float() ) )

        ## return predictions and the model
        return( val_preds, model )
    
    except:
        return(4, model)


def run_metab_predictions(md,
                          df_with_batch, 
                          conq_out, 
                          comb_out,
                          task='Metabolites', 
                          do_clr_transform=False
                         ):
    
    df = df_with_batch.iloc[:, 1:]
    actual_str = batch_weight_feature_and_nbatchpairs_scaling(1e4, df_with_batch)
    results=[]
    for i in range(745): #
        try:
            y=md.iloc[:, i+3] > md.iloc[:, i+3].median()
            print(i)
            inds=df.sum(axis=1) > 10
            comb_out.index = df_with_batch.index

            results_dict = all_classification_pipelines.run_predictions(y.loc[inds], 
                                                                        df_with_batch.loc[inds], 
                                                                        conq_out.loc[inds], 
                                                                        comb_out.loc[inds],
                                                                        do_clr_transform=do_clr_transform,
                                                                        b_str = actual_str,
                                                                        learning_rate = 0.005
                                                                        )
            results.append(results_dict)
        except:
            pass


    plotting.produce_auroc_boxplot(

        pd.concat( [ pd.DataFrame({'auROC': flatten( [ [np.mean(b)] for a,b in results_dict['aurocs'].items()] ), 
                                    'Group': flatten( [ [a]#*len( results_dict['aurocs']['linear'])
                                                          for a in results_dict['aurocs']])
                                       })
                for results_dict in results ]
             ),
                out_path='../results/{}/Metab-auroc-boxplot-linear.pdf'.format(task))
    
    return(None)


def debiasm_training_approach(X_train, 
                              y_train, 
                              X_val,  
                              actual_str=None,
                              df_with_batch=None, 
                              **kwargs):

    dmc = DebiasMClassifier(batch_str = actual_str, 
                            x_val=X_val.values,
                            min_epochs=100
                            )
    dmc.fit(X_train.values, y_train)
    preds = dmc.predict_proba( X_val.values )
    return(preds[:, 1])



def run_melonpann_and_multitask_benchmarks(df_with_batch, 
                                           md):
    
    df = df_with_batch.iloc[:, 1:]

    tmp = md.iloc[:, 3:-6]

    ## adjust column names so we can align to Melonnpann's R-formatted output
    tmp.columns = tmp.columns.str.replace("[()'-,/:]", '.').str.replace(' ', '.').str.replace('-', '.')\
                            .str.replace('[', '.').str.replace(']', '.')
    qq=tmp.columns.values
    qq[tmp.columns.str[0].isin(['.', '1', '2', '3', '4', '5', '6', '7'])] = 'X' + tmp.loc[:,
                            tmp.columns.str[0].isin(['.', '1', '2','3', '4', '5', '6', '7'])
                                                                          ].columns
    tmp.columns = qq 

    tmp = tmp.loc[:, tmp.apply(lambda x: x.nunique()>3, axis=0) ]
    md__=tmp.copy()

    ## read Melonnpann's outputs
    preds =pd.read_csv('Metabolites/melonnpan_out/MelonnPan_Predicted_Metabolites.txt', 
            sep='\t', index_col=0)
    
    
    
    seed=1

    df_val = df.loc[preds.index]
    df_train = df.loc[df.index.isin(preds.index)==False]
    y_train = tmp.loc[df_train.index]
    y_val = tmp.loc[df_val.index]

    np.random.seed(seed)
    torch.manual_seed(seed)
    df_with_batch = pd.concat([pd.Series( pd.Categorical(md.Study).codes, 
                                             index=md.index), df], axis=1) 

    df_train, df_val, yt, yv = df_with_batch.loc[df_train.index], df_with_batch.loc[preds.index],\
                                md__.loc[df_train.index][preds.columns],\
                                md__.loc[preds.index][preds.columns]

    ## define the classification task
    meeds = pd.concat( [yt, yv], axis=0 ).apply(lambda x: x.median() )
    yt, yv = yt>meeds, yv>meeds

    
    ## run multitask DEBIAS-M
    vp1, mod2 = SBCMP_mutlitask_train_and_pred(df_train.values, 
                                               df_val.values, 
                                               yt.values, 
                                               yv.values, 
         batch_sim_strength = batch_weight_feature_and_nbatchpairs_scaling(1e4, df_with_batch),
                                               learning_rate=0.005,
                                               min_epochs=100 ## upping min_epochs here to 
                                                                ## give multitask more time explore
                                 ## Doing the same below in single-task cases for consistency, 
                                  ## although it's less important for that scenario
                                  )
    
    mmmulti_preds1 = pd.DataFrame(
                            torch.cat( [a[:, 1].unsqueeze(-1) for a in mod2.forward( torch.Tensor(df_val.values) ) 
                               ], axis=-1 ).detach().numpy(),
                        index=preds.index, 
                        columns=preds.columns
                    )
    
    base = []
    multitask_debias = []

    for col in yt.columns:
        if col in preds.columns:
            base.append(roc_auc_score( yv[col], preds[col]))
            multitask_debias.append(roc_auc_score( yv[col], mmmulti_preds1[col]))
            
            
    ## run single-task debias-m
    actual_str = batch_weight_feature_and_nbatchpairs_scaling(1e4, df_with_batch)
    singletask_debiasm = []
    for col in yv.columns:
        ppps = debiasm_training_approach(df_train, 
                                         yt[col].values, 
                                         df_val, 
                                         actual_str = actual_str
                                        )
        singletask_debiasm.append(roc_auc_score(yv[col].values, ppps))
    
    
    
    pd.DataFrame( {'Melonnpann':base,
                   'DEBIAS-M Single-task':singletask_debiasm,
                   'DEBIAS-M Multitask':multitask_debias
                  }, 
                 index=preds.columns
                ).to_csv('../results/Metabolites/full-multitask-eval.csv')
    
    plt.figure(figsize=(12,7))
    sns.boxplot(y=base + singletask_debiasm + multitask_debias, 
                x = ['Melonpann']*len(base) + \
                        ['DEBIAS-M Single-task']*len(singletask_debiasm) + \
                        ['DEBIAS-M Multitask']*len(multitask_debias)
               )

    sns.swarmplot(y=base + singletask_debiasm + multitask_debias, 
               x = ['Melonpann']*len(base) + \
                        ['DEBIAS-M Single-task']*len(singletask_debiasm) + \
                        ['DEBIAS-M Multitask']*len(multitask_debias),
                  s=10, 
                  color='k'
               )
    
    
    plt.savefig('../results/Metabolites/full-multitask-eval.pdf', 
                format='pdf', 
                dpi=900
                )
    
    return(None)






