

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.functional import pairwise_distance
from sklearn.base import BaseEstimator

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

class PL_DEBIASM(pl.LightningModule):
    """Logistic regression model."""

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
        use_log: bool=False,
        y=None,
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
        
        
        self.linear = nn.Linear(in_features=self.hparams.input_dim, 
                                out_features=self.hparams.num_classes, 
                                bias=bias)

        self.X=X[:, 1:]
        self.bs=X[:, 0].long()
        
        ### for the 'WITHIN-CONTROL-comparison
        self.X=self.X[y==0]
        self.bs=self.bs[y==0]
        
        self.unique_bs=self.bs.unique().long()
        
        self.n_batches=self.unique_bs.max()+1
        self.batch_weights = torch.nn.Parameter(data = torch.zeros(self.n_batches,
                                                                   input_dim))

        self.batch_sim_str=batch_sim_strength
        
        
    def forward(self, x: Tensor) -> Tensor:
        batch_inds, x = x[:, 0], x[:, 1:]
        x = F.normalize( torch.pow(2, self.batch_weights[batch_inds.long()] ) * x, p=1 )
        x = self.linear(x)
        y_hat = softmax(x)
        return y_hat

    
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch
        
        # flatten any input
        x = x.view(x.size(0), -1)

        y_hat = self.forward(x)
        
        
        # PyTorch cross_entropy function combines log_softmax and nll_loss in single function
        loss = F.cross_entropy(y_hat, y, reduction="sum")
        
        # L1 regularizer
        if self.hparams.l1_strength > 0:
            l1_reg = self.linear.weight.abs().sum()
            loss += self.hparams.l1_strength * l1_reg

        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = self.linear.weight.pow(2).sum()
            loss += self.hparams.l2_strength * l2_reg
            
        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = self.linear.weight.pow(2).sum()
            loss += self.hparams.l2_strength * l2_reg
            
        # DOMAIN similarity regularizer / bias correction
        if self.batch_sim_str > 0:
            x1 = torch.stack( [ ( torch.pow(2, self.batch_weights\
                                          )[torch.where(self.unique_bs==a)[0]] * \
                    (self.X[ torch.where(self.bs==a)[0] ] )  \
                   ).mean(axis=0) for a in self.unique_bs ] )

            x1 = F.normalize(x1, p=1)

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
        y_hat = self.forward(x)
         # PyTorch cross_entropy function combines log_softmax and nll_loss in single function
        loss = F.cross_entropy(y_hat, y, reduction="sum")
        
        # L1 regularizer
        if self.hparams.l1_strength > 0:
            l1_reg = self.linear.weight.abs().sum()
            loss += self.hparams.l1_strength * l1_reg

        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = self.linear.weight.pow(2).sum()
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
    
    
    
    
def DEBIASM_train_and_pred(X_train, 
                         X_val, 
                         y_train, 
                         y_val,
                         batch_sim_strength=1,
                         w_l2 = 0,
                         batch_size=None,
                         learning_rate=0.005,
                         l2_strength=0,
                         includes_batches=False,
                         val_split=0.1, 
                         test_split=0,
                         min_epochs=15, 
                         verbose=False
                         ):
    
    if batch_size is None:
        batch_size=X_train.shape[0]
    
    baseline_mod = LogisticRegression(max_iter=2500)
    baseline_mod.fit(rescale( X_train[:, 1:]), y_train)
        
        
    model = PL_DEBIASM(X = torch.tensor( np.vstack((X_train, X_val)) ),
                       batch_sim_strength = batch_sim_strength,
                       input_dim = X_train.shape[1]-1, 
                       num_classes = 2, 
                       batch_size = batch_size,
                       learning_rate = learning_rate,
                       l2_strength = l2_strength, 
                       w_l2 = w_l2,
                       y=np.hstack((y_train,y_train))
                       )
    
    ## initialize parameters to lbe similar to standard logistic regression
    model.linear.weight.data[0]=-torch.tensor( baseline_mod.coef_[0] )
    model.linear.weight.data[1]= torch.tensor( baseline_mod.coef_[0] )

    ## build pl dataloader
    dm = SklearnDataModule(X_train, 
                           torch.tensor( y_train ).long().detach().numpy(),#due to windows numpy bug
                           val_split=val_split,
                           test_split=test_split
                           )

    ## run training
    trainer = pl.Trainer(logger=False, 
                         checkpoint_callback=False,
                         callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=2)], 
                         check_val_every_n_epoch=2, 
                         weights_summary=None, 
                         progress_bar_refresh_rate=0,#verbose, 
                         min_epochs=min_epochs
                         )
    trainer.fit(model, 
                train_dataloaders=dm.train_dataloader(), 
                val_dataloaders=dm.val_dataloader()
               )
    
    ## get val predictions
    val_preds = model.forward( torch.tensor( X_val ).float() )[:, 1].detach().numpy()
    
    ## return predictions and the model
    return( val_preds, model )

def batch_weight_feature_and_nbatchpairs_scaling(strength, df_with_batch):
    nbs = df_with_batch.iloc[:, 0].nunique()
    w = nbs * (nbs - 1) / 2
    return(strength /( w * ( df_with_batch.shape[1] - 1 ) ) )



class AdaptationDebiasMClassifierWithinControlSimilarity(BaseEstimator):
    """DebiasMClassifier: an sklean-style wrapper for the DEBIAS-M torch implementation."""
    def __init__(self,
                 *, 
                 batch_str = 'infer',
                 learning_rate=0.005, 
                 min_epochs=25,
                 l2_strength=0,
                 w_l2=0,
                 random_state=None
                 ):
        
        self.learning_rate=learning_rate
        self.min_epochs=min_epochs
        self.l2_strength=l2_strength
        self.w_l2=w_l2
        self.batch_str=batch_str
        self.random_state=random_state
        
    def fit(self, X, y, sample_weight=None):
        """Fit the baseline classifier.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if self.batch_str=='infer':
            self.batch_str = batch_weight_feature_and_nbatchpairs_scaling(1e4, pd.DataFrame(X) )
        
        self.classes_ = np.unique(y)
        preds, mod = DEBIASM_train_and_pred(
                                            X, 
                                            X, 
                                            y, 
                                            y,
                                            batch_sim_strength = self.batch_str/2,
                                            learning_rate=self.learning_rate,
                                            min_epochs= self.min_epochs,
                                            l2_strength=self.l2_strength,
                                            w_l2 = self.w_l2
                                            )
        self.model = mod
        self.train_X = pd.DataFrame(X)
        self.train_y = y
        
        return self
    
    def transform(self, X):
        if self.model is None:
            raise(ValueError('You must run the `.fit()` method before executing this transformation'))
            
        if type(X)==pd.DataFrame:
            x = torch.tensor(X.values)
        else:
            x = torch.tensor(X)
            
        batch_inds, x = x[:, 0], x[:, 1:]
        x = F.normalize( torch.pow(2, self.model.batch_weights[batch_inds.long()] ) * x, p=1 )
        
        if type(X)==pd.DataFrame:
            return( pd.DataFrame(x.detach().numpy(), 
                                 index=X.index, 
                                 columns=X.columns[1:]))
        else:
            return( x.detach().numpy() )


    
    def estimate_test_biases(self, x_test, iters=10000):
        
        train_yval_mask=self.train_y == 0
        
        x_base = pd.concat([ self.train_X[train_yval_mask].iloc[:, 0], 
                             self.transform(self.train_X[train_yval_mask])], 
                           axis=1 )

        x_base_means = F.normalize( torch.tensor( x_base.groupby(0)[x_base.columns].mean().iloc[:, 1:].values ),
                                   p=1 )

        input_dim=x_test.shape[1]
        test_bcfs = torch.nn.Parameter(data = torch.zeros(1,
                                                          input_dim)
                                      )

        optimizer=torch.optim.Adam([test_bcfs], 
                                   lr = 0.005
                                   )
        
        x_test_tensor = torch.tensor(x_test)
        
        for i in range(iters):                
            xt1 = F.normalize( x_test_tensor*torch.pow(2, test_bcfs), p=1)
            loss = sum( [pairwise_distance(xt1, a) 
                         for a in x_base_means] ).sum()*0.01 ## really just a lr scaling factor
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        debiased_X_test = F.normalize( x_test_tensor * torch.pow(2, test_bcfs), p=1 )
        return(debiased_X_test)
        

    def predict(self, X):
        """Perform classification on test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted target values for X.
        """
        debiased_X_test = self.estimate_test_biases(X)
        y = F.softmax( self.model.linear(debiased_X_test), dim=1 )[:, 1].detach().numpy() > 0.5
        return y

    def predict_proba(self, X):
        """
        Return probability estimates for the test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        P : ndarray of shape (n_samples, n_classes) or list of such arrays
            Returns the probability of the sample for each class in
            the model, where classes are ordered arithmetically, for each
            output.
        """
        debiased_X_test = self.estimate_test_biases(X)
        P = F.softmax( self.model.linear(debiased_X_test), dim=1 ).detach().numpy()
        return P