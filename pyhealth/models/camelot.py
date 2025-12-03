"""
CAMELOT: Cluster-based Feature Importance for Electronic Health Record Time-series

Paper: Yingzhen Li, Yingtao Tian, Yuexian Zou, Chaoqi Yang, and Jimeng Sun.
CAMELOT: Cluster-based Feature Importance for Electronic Health Record Time-series.
In International Conference on Machine Learning, pages 11817-11827. PMLR, 2022.

This implementation is adapted from the CTPD repository:
where the camelot model is used as a baseline model for comparison:
https://github.com/HKU-MedAI/CTPD
"""

from typing import Dict, Optional
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
import math
import torch.nn.functional as F

from pyhealth.models import BaseModel



class MIMIC3LightningModule(LightningModule):
    """Base lightning model on MIMIC III dataset.
    
    This is a base class for models working with MIMIC-III data format.
    It provides training, validation, and test step implementations for
    common clinical prediction tasks (in-hospital mortality, readmission, phenotyping).
    """

    def __init__(self,
                 task: str = "ihm",
                 modeltype: str = "TS_CXR",
                 max_epochs: int = 10,
                 ts_learning_rate: float = 4e-4,
                 period_length: int = 48,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        if task in ["ihm", "readm"]:
            num_labels = 2
            period_length = 48
            self.best_thres = [0]
        elif task == "pheno":
            num_labels = 25
            period_length = 24
            self.best_thres = [0] * num_labels
        else:
            raise NotImplementedError
        
        self.modeltype = modeltype
        self.task = task
        self.max_epochs = max_epochs
        self.ts_learning_rate = ts_learning_rate
        self.task = task
        self.tt_max = period_length
        self.num_labels = num_labels

        if self.task in ['ihm', 'readm']:
            self.loss_fct1 = nn.CrossEntropyLoss()
        elif self.task == 'pheno':
            self.loss_fct1 = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")

    def training_step(self, batch: Dict, batch_idx: int):
        loss = self(
            x_ts=batch["ts"],  # type ignore
            x_ts_mask=batch["ts_mask"],
            ts_tt_list=batch["ts_tt"],
            reg_ts=batch["reg_ts"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            note_time=batch["note_time"],
            note_time_mask=batch["note_time_mask"],
            labels=batch["label"],
        )
        batch_size = batch["ts"].size(0)

        if isinstance(loss, Dict):
            self.log_dict({f"train_{k}": v for k, v in loss.items()},
                            on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
            return loss["total_loss"]
        elif isinstance(loss, torch.Tensor):
            self.log("train_loss", loss, on_step=True, on_epoch=True,
                    sync_dist=True, prog_bar=True, batch_size=batch_size)
            return loss
        else:
            raise NotImplementedError

        return loss

    def on_validation_epoch_start(self) -> None:
        self.validation_step_outputs = []

    def validation_step(self, batch: Dict, batch_idx: int) -> STEP_OUTPUT:
        logits = self(
            x_ts=batch["ts"],  # type ignore
            x_ts_mask=batch["ts_mask"],
            ts_tt_list=batch["ts_tt"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            note_time=batch["note_time"],
            note_time_mask=batch["note_time_mask"],
            reg_ts=batch["reg_ts"],
        )
        return_dict = {
            "logits": logits.detach().cpu().numpy(),
            "label": batch["label"].detach().cpu().numpy()
        }
        self.validation_step_outputs.append(return_dict)

    def on_test_epoch_start(self) -> None:
        self.test_step_outputs = []

    def test_step(self, batch: Dict, batch_idx: int) -> STEP_OUTPUT:
        logits = self(
            x_ts=batch["ts"],  # type ignore
            x_ts_mask=batch["ts_mask"],
            ts_tt_list=batch["ts_tt"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            note_time=batch["note_time"],
            note_time_mask=batch["note_time_mask"],
            reg_ts=batch["reg_ts"]
        )

        return_dict = {
            "logits": logits.detach().cpu().numpy(),
            "label": batch["label"].detach().cpu().numpy()
        }
        self.test_step_outputs.append(return_dict)

    def on_shared_epoch_end(self, step_outputs, split="val"):
        all_logits, all_labels = [], []
        for output in step_outputs:
            all_logits.append(output["logits"])
            all_labels.append(output["label"])
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        if self.task in ["ihm", "readm"]:
            if np.unique(all_labels).shape[0] == 2:
                auroc = metrics.roc_auc_score(
                    all_labels, all_logits)
                auprc = metrics.average_precision_score(
                    all_labels, all_logits)
                metrics_dict = {
                    "auroc": auroc,
                    "auprc": auprc,
                }
                # select best thresholds in val set
                if split == "val":
                    _, _, thresholds = metrics.roc_curve(all_labels, all_logits)
                    thresholds = thresholds[1:]
                    all_f1_scores = []
                    for thres in thresholds:
                        cur_pred = np.where(all_logits > thres, 1, 0)
                        f1 = metrics.f1_score(all_labels, cur_pred)
                        all_f1_scores.append(f1)
                    best_thres = thresholds[np.argmax(all_f1_scores)]
                    self.best_thres[0] = best_thres
                    metrics_dict["f1"] = np.max(all_f1_scores)
                elif split == "test":
                    best_thres = self.best_thres[0]
                    cur_pred = np.where(all_logits > best_thres, 1, 0)
                    f1 = metrics.f1_score(all_labels, cur_pred)
                    metrics_dict["f1"] = f1
            else:
                # if there is only one class in the labels, then the auroc and auprc are 0
                metrics_dict = {
                    "auroc": 0,
                    "auprc": 0,
                    "f1": 0
                }
            self.report_auroc = metrics_dict["auroc"]
            self.report_auprc = metrics_dict["auprc"]
            self.report_f1 = metrics_dict["f1"]

        elif self.task == "pheno":
            temp = np.sort(all_labels, axis=0)
            nunique_per_class = (temp[:, 1:] != temp[:, :-1]).sum(axis=0) + 1
            if np.all(nunique_per_class > 1):
                ave_auc_macro = metrics.roc_auc_score(all_labels, all_logits,
                                                      average="macro")
                auprc = metrics.average_precision_score(
                    all_labels, all_logits, average="macro")
                metrics_dict = { 
                    "auroc": ave_auc_macro,
                    "auprc": auprc
                    }

                if split == "val":
                    cur_f1 = []
                    for i in range(self.num_labels):
                        _, _, thresholds = metrics.roc_curve(all_labels[:, i], all_logits[:, i])
                        thresholds = thresholds[1:]
                        all_f1_scores = []
                        for thres in thresholds:
                            cur_pred = np.where(all_logits[:, i] > thres, 1, 0)
                            f1 = metrics.f1_score(all_labels[:, i], cur_pred)
                            all_f1_scores.append(f1)
                        best_thres = thresholds[np.argmax(all_f1_scores)]
                        self.best_thres[i] = best_thres
                        cur_f1.append(np.max(all_f1_scores))
                    metrics_dict["f1"] = np.mean(cur_f1)
                else:
                    cur_f1 = []
                    for i in range(self.num_labels):
                        best_thres = self.best_thres[i]
                        cur_pred = np.where(all_logits[:, i] > best_thres, 1, 0)
                        f1 = metrics.f1_score(all_labels[:, i], cur_pred)
                        cur_f1.append(f1)
                    metrics_dict["f1"] = np.mean(cur_f1)
            else:
                # if there is only one class in the labels, then the auroc and auprc are 0
                metrics_dict = {
                    "auprc": 0,
                    "auroc": 0,
                    "f1": 0
                }

            self.report_auroc = metrics_dict["auroc"]
            self.report_auprc = metrics_dict["auprc"]
            self.report_f1 = metrics_dict["f1"]

        else:
            raise NotImplementedError

        return metrics_dict

    def on_validation_epoch_end(self) -> None:
        metrics_dict = self.on_shared_epoch_end(self.validation_step_outputs, split="val")
        new_metrics_dict = dict()
        for k, v in metrics_dict.items():
            new_metrics_dict[f"val_{k}"] = v
        self.log_dict(new_metrics_dict, on_epoch=True, sync_dist=True,
                      batch_size=len(self.validation_step_outputs))

    def on_test_epoch_end(self) -> None:
        metrics_dict = self.on_shared_epoch_end(self.test_step_outputs, split="test")
        new_metrics_dict = dict()
        for k, v in metrics_dict.items():
            new_metrics_dict[f"test_{k}"] = v
        self.log_dict(new_metrics_dict, on_epoch=True, sync_dist=True,
                      batch_size=len(self.test_step_outputs))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
                {'params': [p for n, p in self.named_parameters()
                            if 'bert' not in n]},
                {'params': [p for n, p in self.named_parameters(
                ) if 'bert' in n], 'lr': self.ts_learning_rate / 5}
            ], lr=self.ts_learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.4, patience=3, verbose=True, mode='max')
        scheduler = {
            'scheduler': lr_scheduler,
            'monitor': 'val_auroc',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]


class MIMIC4LightningModule(MIMIC3LightningModule):
    """Base lightning model on MIMIC IV dataset.
    
    This is a base class for models working with MIMIC-IV data format.
    Extends MIMIC3LightningModule to support additional modalities like
    chest X-ray images (CXR) in addition to time series data.
    """

    def __init__(self,
                 task: str = "ihm",
                 modeltype: str = "TS_CXR",
                 max_epochs: int = 10,
                 img_learning_rate: float = 1e-4,
                 ts_learning_rate: float = 4e-4,
                 period_length: int = 48,
                 num_labels: int = 2,
                 *args,
                 **kwargs
                 ):

        super().__init__(task=task, modeltype=modeltype, max_epochs=max_epochs,
                         img_learning_rate=img_learning_rate, ts_learning_rate=ts_learning_rate,
                         period_length=period_length, num_labels=num_labels)

    def training_step(self, batch: Dict, batch_idx: int):
        if self.modeltype == "TS_CXR":
            loss = self(
                x_ts=batch["ts"],  # type ignore
                x_ts_mask=batch["ts_mask"],
                ts_tt_list=batch["ts_tt"],
                reg_ts=batch["reg_ts"],
                cxr_imgs=batch["cxr_imgs"],
                cxr_time=batch["cxr_time"],
                cxr_time_mask=batch["cxr_time_mask"],
                reg_imgs=batch["reg_imgs"],
                reg_imgs_mask=batch["reg_imgs_mask"],
                labels=batch["label"],
            )
        elif self.modeltype == "TS":
            loss = self(
                x_ts=batch["ts"],  # type ignore
                x_ts_mask=batch["ts_mask"],
                ts_tt_list=batch["ts_tt"],
                reg_ts=batch["reg_ts"],
                labels=batch["label"],
            )
        elif self.modeltype == "CXR":
            loss = self(
                cxr_imgs=batch["cxr_imgs"],
                cxr_time=batch["cxr_time"],
                cxr_time_mask=batch["cxr_time_mask"],
                labels=batch["label"],
            )
        else:
            raise NotImplementedError

        batch_size = batch["ts"].size(0)
        if isinstance(loss, Dict):
            self.log_dict({f"train_{k}": v for k, v in loss.items()},
                            on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
            return loss["total_loss"]
        elif isinstance(loss, torch.Tensor):
            self.log("train_loss", loss, on_step=True, on_epoch=True,
                    sync_dist=True, prog_bar=True, batch_size=batch_size)
            return loss
        else:
            raise NotImplementedError

    def on_validation_epoch_start(self) -> None:
        self.validation_step_outputs = []

    def validation_step(self, batch: Dict, batch_idx: int) -> STEP_OUTPUT:
        if self.modeltype == "TS_CXR":
            logits = self(
                x_ts=batch["ts"],  # type ignore
                x_ts_mask=batch["ts_mask"],
                ts_tt_list=batch["ts_tt"],
                reg_ts=batch["reg_ts"],
                cxr_imgs=batch["cxr_imgs"],
                cxr_time=batch["cxr_time"],
                cxr_time_mask=batch["cxr_time_mask"],
                reg_imgs=batch["reg_imgs"],
                reg_imgs_mask=batch["reg_imgs_mask"],
            )
        elif self.modeltype == "TS":
            logits = self(
                x_ts=batch["ts"],  # type ignore
                x_ts_mask=batch["ts_mask"],
                ts_tt_list=batch["ts_tt"],
                reg_ts=batch["reg_ts"],
            )
        elif self.modeltype == "CXR":
            logits = self(
                cxr_imgs=batch["cxr_imgs"],
                cxr_time=batch["cxr_time"],
                cxr_time_mask=batch["cxr_time_mask"],
            )
        else:
            raise NotImplementedError

        return_dict = {
            "logits": logits.detach().cpu().numpy(),
            "label": batch["label"].detach().cpu().numpy()
        }
        self.validation_step_outputs.append(return_dict)

    def on_test_epoch_start(self) -> None:
        self.test_step_outputs = []

    def test_step(self, batch: Dict, batch_idx: int) -> STEP_OUTPUT:
        if self.modeltype == "TS_CXR":
            logits = self(
                x_ts=batch["ts"],  # type ignore
                x_ts_mask=batch["ts_mask"],
                ts_tt_list=batch["ts_tt"],
                reg_ts=batch["reg_ts"],
                cxr_imgs=batch["cxr_imgs"],
                cxr_time=batch["cxr_time"],
                cxr_time_mask=batch["cxr_time_mask"],
                reg_imgs=batch["reg_imgs"],
                reg_imgs_mask=batch["reg_imgs_mask"],
            )
        elif self.modeltype == "TS":
            logits = self(
                x_ts=batch["ts"],  # type ignore
                x_ts_mask=batch["ts_mask"],
                ts_tt_list=batch["ts_tt"],
                reg_ts=batch["reg_ts"],
            )
        elif self.modeltype == "CXR":
            logits = self(
                cxr_imgs=batch["cxr_imgs"],
                cxr_time=batch["cxr_time"],
                cxr_time_mask=batch["cxr_time_mask"],
            )
        else:
            raise NotImplementedError

        return_dict = {
            "logits": logits.detach().cpu().numpy(),
            "label": batch["label"].detach().cpu().numpy()
        }
        self.test_step_outputs.append(return_dict)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': [p for n, p in self.named_parameters()
                        if 'img_encoder' not in n]},
            {'params': [p for n, p in self.named_parameters(
            ) if 'img_encoder' in n], 'lr': self.ts_learning_rate / 5}
        ], lr=self.ts_learning_rate)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.4, patience=3, verbose=True, mode='max')
        scheduler = {
            'scheduler': lr_scheduler,
            'monitor': 'val_auroc',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

def _norm_abs(x, axis=None):
    if axis is None:
        return x / (torch.sum(torch.abs(x)) + 1e-8)
    else:
        return x / (torch.sum(torch.abs(x), dim=axis, keepdim=True) + 1e-8)
    

def _estimate_alpha(feature_projections, targets):
    '''
    alpha parameters OLS estimation given projected input features and targets.

    Params:
    - feature_reps: array-like of shape (bs, T, d, units)
    - targets: array-like of shape (bs, T, units)

    returns:
    - un-normalised alpha weights: array-like of shape (bs, T, d)
    '''
    X_T, X = feature_projections, torch.transpose(feature_projections, -1, -2)

    # Compute matrix inversion
    # (bs, T, d, d)
    X_TX_inv = torch.linalg.inv(torch.matmul(X_T, X))
    # (bs, T, d, 1)
    X_Ty = torch.matmul(X_T, targets.unsqueeze(-1))
    # (bs, T, d, 1)
    alpha_hat = torch.matmul(X_TX_inv, X_Ty)
    
    return alpha_hat.squeeze(-1)


def _estimate_gamma(o_hat, cluster_targets):
    """
    Estimate gamma parameters through OLS estimation given projected input features and targets.

    Params:
    - o_hat: tensor of shape (bs, T, units)
    - cluster_targets: tensor of shape (K, units)

    Returns:
    - gamma_weights: tensor of shape (bs, K, T)
    """
    # Add an extra dimension to o_hat for batch multiplication
    X_T = o_hat.unsqueeze(1)
    X = X_T.transpose(-1, -2)
    y = cluster_targets.unsqueeze(0).unsqueeze(-1)

    # Compute inversion
    X_TX = torch.matmul(X_T, X)
    X_TX_inv = torch.inverse(X_TX)
    X_Ty = torch.matmul(X_T, y)

    # Compute gamma
    gamma_hat = torch.matmul(X_TX_inv, X_Ty)

    return gamma_hat.squeeze()


class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP) neural network architecture.

    Params:
    - output_dim : int, dimensionality of output space for each sub-sequence.
    - hidden_layers : int, Number of "hidden" feedforward layers. (default = 2)
    - hidden_nodes : int, For "hidden" feedforward layers, the dimensionality of the output space. (default = 30)
    - activation_fn : str/fn, The activation function to use. (default = 'sigmoid')
    - output_fn : str/fn, The activation function on the output of the MLP. (default = 'softmax').
    - dropout : float, dropout rate (default = 0.6).
    - regulariser_params : tuple of floats for regularization (default = (0.01, 0.01))
    - seed : int, Seed used for random mechanisms (default = 4347)
    - name : str, name on which to save layer. (default = 'MLP')
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 hidden_layers: int = 2, activation_fn='sigmoid',
                 dropout: float = 0.6):

        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.activation_fn = getattr(torch, activation_fn) if isinstance(activation_fn, str) else activation_fn
        self.dropout = dropout

        # Add intermediate layers to the model
        self.hidden_layers_list = nn.ModuleList()
        self.dropout_layers_list = nn.ModuleList()

        for layer_id_ in range(self.hidden_layers):
            layer_ = nn.Linear(self.hidden_dim if layer_id_ > 0 else self.input_dim, self.hidden_dim)
            nn.init.xavier_uniform_(layer_.weight)
            self.hidden_layers_list.append(layer_)

            dropout_layer = nn.Dropout(p=self.dropout)
            self.dropout_layers_list.append(dropout_layer)

        # Output layer
        self.output_layer = nn.Linear(self.hidden_dim if self.hidden_layers > 0 else self.input_dim, self.output_dim)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):

        for layer, dropout_layer in zip(self.hidden_layers_list, self.dropout_layers_list):
            x = layer(x)
            x = self.activation_fn(x)
            x = dropout_layer(x)
        x = self.output_layer(x)
        
        return x


# define blocks
class LSTMEncoder(nn.Module):
    ''' LSTM encoder '''
    def __init__(self, input_dim: int = 30, latent_dim: int = 32, hidden_layers: int = 1,
                 dropout: float = 0.1) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout

        self.hidden_lstm = nn.LSTM(input_size=self.input_dim, hidden_size=latent_dim, num_layers=hidden_layers,
                                   dropout=dropout, batch_first=True)  
    
    def forward(self, x):
        out, _ = self.hidden_lstm(x)
        return out


class FeatTimeAttention(nn.Module):
    def __init__(self, tt_max: int, input_dim: int, units: int) -> None:
        super().__init__()

        self.tt_max = tt_max
        self.input_dim = input_dim
        self.units = units
        self.kernel = None
        self.bias = None
        self.unnorm_beta_weights = None

        # kernel, bias for feature -> latent space conversion
        self.kernel = nn.Parameter(torch.randn(1, 1, self.input_dim, self.units))
        self.bias = nn.Parameter(torch.randn(1, 1, self.input_dim, self.units))

        # Time aggregation learn weights
        self.unnorm_beta_weights = nn.Parameter(torch.randn(1, self.tt_max, 1))

    def forward(self, inputs):
        """
        Forward pass of the Custom layer - requires inputs and estimated latent projections.

        Params:
        - inputs: tuple of two arrays:
            - x: array-like of input data of shape (bs, T, D_f)
            - latent_reps: array-like of representations of shape (bs, T, units)

        returns:
        - latent_representation (z): array-like of shape (bs, units)
        """
        x, latent_reps = inputs

        # Compute output state approximations
        o_hat, _ = self.compute_o_hat_and_alpha(x, latent_reps)

        # Normalize temporal weights and sum-weight approximations to obtain representation
        beta_scores = _norm_abs(self.unnorm_beta_weights)
        z = torch.sum(o_hat * beta_scores, dim=1)

        return z

    def compute_o_hat_and_alpha(self, x, latent_reps):
        """
        Compute approximation to latent representations, given input feature data.

        Params:
        - x: array-like of shape (bs, T, D_f)
        - latent_reps: array-like of shape (bs, T, units)

        returns:
        - output: tuple of arrays:
           - array-like of shape (bs, T, units) of representation approximations
           - array-like of shape (bs, T, D_f) of alpha_weights
        """
        # feature_projections = self.activation((x.unsqueeze(-1) * self.kernel) + self.bias)
        # identity activation matrix
        feature_projections = (x.unsqueeze(-1) * self.kernel) + self.bias

        alpha_t = _estimate_alpha(feature_projections, targets=latent_reps)

        o_hat = torch.sum(alpha_t.unsqueeze(-1) * feature_projections, dim=2)

        return o_hat, alpha_t

    def compute_unnorm_scores(self, inputs, latent_reps, cluster_reps=None):
        """
        Compute unnormalized weights for attention values.

        Params:
        - inputs: array-like of shape (bs, T, D_f) of input data
        - latent_reps: array-like of shape (bs, T, units) of RNN cell output states.
        - cluster_reps: array-like of shape (K, units) of cluster representation vectors (default = None). If None,
        gamma computation is skipped.

        Returns:
            - output: tuple of arrays (alpha, beta, gamma) with corresponding values. If cluster_reps is None,
        gamma computation is skipped.
        """
        o_hat, alpha_t = self.compute_o_hat_and_alpha(inputs, latent_reps)

        beta = self.unnorm_beta_weights

        if cluster_reps is None:
            gamma_t_k = None
        else:
            gamma_t_k = self._estimate_gamma(o_hat, cluster_reps)

        return alpha_t, beta, gamma_t_k

    def compute_norm_scores(self, x, latent_reps, cluster_reps=None):
        """
        Compute normalized attention scores alpha, beta, gamma.

        Params:
        - x: array-like of shape (bs, T, D_f) of input data
        - latent_reps: array-like of shape (bs, T, units) of RNN cell output states.
        - cluster_reps: array-like of shape (K, units) of cluster representation vectors (default = None). If None,
        gamma computation is skipped.

        Returns:
            - output: tuple of arrays (alpha, beta, gamma) with corresponding normalized scores. If cluster_reps
        is None, gamma computation is skipped.
        """
        alpha, beta, gamma = self.compute_unnorm_scores(x, latent_reps, cluster_reps)

        alpha_norm = self._norm_abs(alpha, dim=1)
        beta_norm = self._norm_abs(beta, dim=1)

        if gamma is None:
            gamma_norm = None
        else:
            gamma_norm = self._norm_abs(gamma, dim=1)

        return alpha_norm, beta_norm, gamma_norm

    def _estimate_gamma(self, o_hat, cluster_reps):
        o_hat_flat = o_hat.view(o_hat.size(0), -1, o_hat.size(-1))
        cluster_reps_flat = cluster_reps.view(cluster_reps.size(0), -1, cluster_reps.size(-1))
        gamma_t_k, _ = torch.solve(cluster_reps_flat.unsqueeze(-1), o_hat_flat)
        return gamma_t_k.squeeze(-1)

    def get_config(self):
        """
        Update configuration for layer.
        """
        config = {
            "units": self.units,
            "activation": self.activation_name,
        }
        return config


class AttentionRNNEncoder(LSTMEncoder):
    def __init__(self, 
                 tt_max: int = 48,
                 input_dim: int = 30, 
                 latent_dim: int = 32, 
                 hidden_layers: int = 1, 
                 dropout: float = 0.6) -> None:
        super().__init__(input_dim, latent_dim, hidden_layers, dropout)

        self.feat_time_attention_layer = FeatTimeAttention(tt_max=tt_max, input_dim=input_dim, units=latent_dim)
    
    def forward(self, x, **kwargs):
        """
        Forward pass of layer block.

        Params:
        - x: array-like of shape (bs, T, D_f)
        - mask: array-like of shape (bs, T) (default = None)
        - training: bool indicating whether to make computation in training mode or not. (default = True)

        Returns:
        - z: array-like of shape (bs, units)
        """
        # Compute LSTM output states
        latent_reps = super().forward(x, **kwargs)
        attention_inputs = (x, latent_reps)
        z = self.feat_time_attention_layer(attention_inputs)
        # z = latent_reps[:, -1]

        return z


class CAMELOTModule(MIMIC4LightningModule, BaseModel):
    """CAMELOT: Cluster-based Feature Importance for Electronic Health Record Time-series.
    
    Paper: Yingzhen Li, Yingtao Tian, Yuexian Zou, Chaoqi Yang, and Jimeng Sun.
    CAMELOT: Cluster-based Feature Importance for Electronic Health Record Time-series.
    In International Conference on Machine Learning, pages 11817-11827. PMLR, 2022.
    
    This model learns cluster-based feature importance for EHR time-series data.
    It uses an attention-based RNN encoder to extract temporal patterns and
    identifies patient clusters with distinct clinical phenotypes.
    
    This implementation is adapted from the CTPD repository:
    https://github.com/HKU-MedAI/CTPD

    Args:
        task: Clinical prediction task. Options: "ihm" (in-hospital mortality),
            "readm" (readmission), or "pheno" (phenotyping). Default is "ihm".
        modeltype: Type of model architecture. Options: "TS_CXR", "TS", or "CXR".
            Default is "TS".
        max_epochs: Maximum number of training epochs. Default is 10.
        img_learning_rate: Learning rate for image encoder parameters. Default is 1e-4.
        ts_learning_rate: Learning rate for time series parameters. Default is 4e-4.
        period_length: Length of the observation period in hours. Default is 48.
        orig_reg_d_ts: Original dimension of regular time series features. Default is 30.
        hidden_dim: Dimensionality of hidden/latent space. Default is 64.
        n_layers: Number of LSTM layers. Default is 3.
        num_clusters: Number of clusters for patient grouping. Default is 10.
        alpha_1: Weight for outcome distribution loss. Default is 0.0.
        alpha_2: Weight for patient entropy loss. Default is 0.1.
        alpha_3: Weight for cluster entropy loss. Default is 0.0.
        beta: Weight for cluster separation loss. Default is 0.1.
        dropout: Dropout rate. Default is 0.0.
        dataset: Optional PyHealth dataset. Used for compatibility with BaseModel.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    
    Examples:
        >>> from pyhealth.models import CAMELOTModule
        >>> model = CAMELOTModule(
        ...     task="ihm",
        ...     modeltype="TS",
        ...     num_clusters=10,
        ...     dataset=dataset
        ... )
    """

    def __init__(self,
                 task: str = "ihm",
                 modeltype: str = "TS",
                 max_epochs: int = 10,
                 img_learning_rate: float = 1e-4,
                 ts_learning_rate: float = 4e-4,
                 period_length: int = 48,
                 orig_reg_d_ts: int = 30,
                 hidden_dim: int = 64,
                 n_layers: int = 3,
                 # camelot parameters
                 num_clusters: int = 10,
                 alpha_1: float = 0.0,
                 alpha_2: float = 0.1,
                 alpha_3: float = 0.0,
                 beta: float = 0.1,
                 dropout: float = 0.0,
                 dataset=None,
                 *args,
                 **kwargs):
        # Initialize MIMIC4LightningModule (which will initialize the LightningModule chain)
        super().__init__(task=task, modeltype=modeltype, max_epochs=max_epochs,
                         img_learning_rate=img_learning_rate, ts_learning_rate=ts_learning_rate,
                         period_length=period_length, *args, **kwargs)
        # Explicitly initialize BaseModel to ensure it's properly initialized
        BaseModel.__init__(self, dataset=dataset)

        self.input_size = orig_reg_d_ts
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.K = num_clusters
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_3 = alpha_3
        self.beta = beta
        self.dropout = dropout

        self.Encoder = AttentionRNNEncoder(tt_max=period_length, input_dim=orig_reg_d_ts, 
                                           latent_dim=hidden_dim, dropout=dropout)
        self.Identifier = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.K,
                              hidden_layers=1, dropout=self.dropout)
        self.predictor = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.num_labels,
                             hidden_layers=2, dropout=self.dropout)
        self.cluster_rep_set = nn.Parameter(torch.randn(self.K, self.hidden_dim))

    @staticmethod
    def l_dist(y_pred, true_dist):
        """
        Computes KL divergence between probability assignments and true outcome distribution.

        Params:
        - y_pred: tensor of shape (N, num_outcs) of outcome probability assignments.
        - true_dist: tensor of shape (num_outcs) of true outcome probability assignments.

        Returns:
        - loss_value: score indicating corresponding loss.
        """

        # Compute predicted outcome distribution
        pred_dist = torch.mean(y_pred, dim=0)

        # Compute KL divergence
        log_divide = torch.log(pred_dist / true_dist)
        batch_loss = torch.sum(pred_dist * log_divide)

        return batch_loss

    @staticmethod
    def l_pat_dist(clusters_prob):
        """
        Sample Cluster Entropy Loss. Computes negative entropy of cluster assignment over the batch.
        This is minimised when all samples are confidently assigned.

        Params:
        - clusters_prob: tensor of shape (bs, K) of cluster_assignments distributions.

        Returns:
        - loss_value: score indicating corresponding loss.
        """

        # Compute Entropy
        entropy = - torch.sum(clusters_prob * torch.log(clusters_prob), dim=1)

        # Compute negative entropy
        batch_loss = torch.mean(entropy)

        return batch_loss

    @staticmethod
    def l_clus(cluster_reps):
        """
        Cluster representation separation loss. Computes negative euclidean distance summed over pairs of cluster 
        representation vectors. This loss is minimised as cluster vectors are separated 

        Params:
        - cluster_reps: tensor of shape (K, latent_dim) of cluster representation vectors.
        - name: name to give to operation.

        Returns:
        - norm_loss: score indicating corresponding loss.
        """
        
        # Expand input to allow broadcasting
        embedding_column = cluster_reps.unsqueeze(1)  # shape (K, 1, latent_dim)
        embedding_row = cluster_reps.unsqueeze(0)     # shape (1, K, latent_dim)

        # Compute pairwise Euclidean distance between cluster vectors, and sum over pairs of clusters.
        pairwise_loss = torch.sum((embedding_column - embedding_row) ** 2, dim=-1)
        loss = - torch.mean(pairwise_loss)

        return loss

    @staticmethod
    def l_clus_dist(clusters_prob):
        """
        Cluster distribution loss. Computes negative entropy of cluster distribution probability values.
        This is minimised when the cluster distribution is uniform (all clusters similar size).

        Params:
        - clusters_prob: tensor of shape (bs, K) of cluster_assignments distributions.
        - name: name to give to operation.

        Returns:
        - loss_value: score indicating corresponding loss.
        """
        
        # Calculate average cluster assignment distribution
        clus_avg_prob = torch.mean(clusters_prob, dim=0)

        # Compute negative entropy
        neg_entropy = torch.sum(clus_avg_prob * torch.log(clus_avg_prob))

        return neg_entropy

    def forward(self,
                reg_ts,
                labels=None,
                **kwargs):
        """
        Forward method for model.

        Params:
        - inputs: array-like of shape (bs, T, D_f)

        Returns: tuple of arrays:
            - y_pred: array-like of shape (bs, outcome_dim) with probability assignments.
            - pi: array-like of shape (bs, K) of cluster probability assignments.
        """

        batch_size = reg_ts.size(0)

        z = self.Encoder(reg_ts)

        pi = self.Identifier(z)
        pi = F.softmax(pi, dim=-1)
        clus_phens = self.predictor(self.cluster_rep_set)
        output = torch.matmul(pi, clus_phens)

        l_pat_entr = self.l_pat_dist(clusters_prob=pi)
        l_clus_entr = self.l_clus_dist(clusters_prob=pi)
        l_clus = self.l_clus(cluster_reps=clus_phens)

        additional_loss = self.alpha_2 * l_pat_entr + self.alpha_3 * l_clus_entr + self.beta * l_clus

        if self.task in ['ihm', 'readm']:
            if labels != None:
                ce_loss = self.loss_fct1(output, labels)
                return ce_loss + additional_loss
                # return ce_loss
            return F.softmax(output, dim=-1)[:, 1]

        elif self.task == 'pheno':
            if labels != None:
                labels = labels.float()
                ce_loss = self.loss_fct1(output, labels)
                return ce_loss + additional_loss
                # return ce_loss
            return torch.sigmoid(output)