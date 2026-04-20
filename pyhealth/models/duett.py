from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import x_transformers

from pyhealth.models.base_model import BaseModel
from pyhealth.datasets import SampleDataset


class BatchNormLastDim(nn.Module):
    """Applies BatchNorm1d to the last dimension of higher order tensors."""
    def __init__(self, d: int, **kwargs):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(d, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return self.batch_norm(x)
        elif x.ndim == 3:
            return self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
        else:
            raise NotImplementedError("BatchNormLastDim not implemented for ndim > 3")


def simple_mlp(
    d_in: int,
    d_out: int,
    n_hidden: int,
    d_hidden: int,
    final_activation: bool = False,
    input_batch_norm: bool = False,
    hidden_batch_norm: bool = False,
    dropout: float = 0.0,
    activation: type = nn.ReLU,
) -> nn.Sequential:
    """Builds a sequence of fully connected blocks."""
    layers = []
    if n_hidden == 0:
        if input_batch_norm:
            layers.append(BatchNormLastDim(d_in))
        layers.append(nn.Linear(d_in, d_out))
    else:
        if input_batch_norm:
            layers.append(BatchNormLastDim(d_in))
        layers.extend([nn.Linear(d_in, d_hidden), activation(), nn.Dropout(dropout)])
        for _ in range(n_hidden - 1):
            if hidden_batch_norm:
                layers.append(BatchNormLastDim(d_hidden))
            layers.extend(
                [nn.Linear(d_hidden, d_hidden), activation(), nn.Dropout(dropout)]
            )
        if hidden_batch_norm:
            layers.append(BatchNormLastDim(d_hidden))
        layers.append(nn.Linear(d_hidden, d_out))
        
    if final_activation:
        layers.append(activation())
    return nn.Sequential(*layers)


class DuETTCore(nn.Module):
    """Core Transformer block integrating orthogonal attentions for DuETT.
    
    Operates by alternately processing variables across temporal sequences 
    and feature classes.
    
    Args:
        d_static_num (int): Dimension of static features.
        d_time_series_num (int): Number of time-series features.
        d_target (int): Output dimension.
        d_embedding (int): Hidden embedding scale.
        d_feedforward (int): Feedforward dimension in transformers.
        n_transformer_head (int): Number of attention heads.
        n_duett_layers (int): Number of crossed transformer layers.
        d_hidden_tab_encoder (int): Hidden dimension for tabular encoder.
        n_hidden_tab_encoder (int): Number of hidden layers for tabular encoder.
        n_hidden_head (int): Number of hidden layers for output head.
        d_hidden_head (int): Hidden dimension for output head.
        transformer_dropout (float): Dropout rate for transformers.
        masked_transform_timesteps (int): Number of time bins.
    """
    def __init__(
        self,
        d_static_num: int,
        d_time_series_num: int,
        d_target: int,
        d_embedding: int = 24,
        d_feedforward: int = 512,
        n_transformer_head: int = 2,
        n_duett_layers: int = 2,
        d_hidden_tab_encoder: int = 128,
        n_hidden_tab_encoder: int = 1,
        n_hidden_head: int = 1,
        d_hidden_head: int = 64,
        transformer_dropout: float = 0.0,
        masked_transform_timesteps: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.d_embedding = d_embedding
        self.register_buffer("REPRESENTATION_EMBEDDING_KEY", torch.tensor(1))

        self.special_embeddings = nn.Embedding(8, d_embedding)
        self.embedding_layers = nn.ModuleList([
            simple_mlp(2, d_embedding, 1, 64, hidden_batch_norm=True)
            for _ in range(d_time_series_num)
        ])

        self.n_obs_embedding = nn.Embedding(16, 1)

        et_dim = d_embedding * (masked_transform_timesteps + 1)
        tt_dim = d_embedding * (d_time_series_num + 1)

        self.event_transformers = nn.ModuleList([
            x_transformers.Encoder(
                dim=et_dim, depth=1, heads=n_transformer_head,
                attn_dim_head=d_embedding // n_transformer_head,
                use_scalenorm=True, ff_mult=d_feedforward / et_dim,
                attn_dropout=transformer_dropout, ff_dropout=transformer_dropout,
            )
            for _ in range(n_duett_layers)
        ])

        self.full_event_embedding = nn.Embedding(d_time_series_num + 1, et_dim)

        self.time_transformers = nn.ModuleList([
            x_transformers.Encoder(
                dim=tt_dim, depth=1, heads=n_transformer_head,
                attn_dim_head=d_embedding // n_transformer_head,
                use_scalenorm=True, ff_mult=d_feedforward / tt_dim,
                attn_dropout=transformer_dropout, ff_dropout=transformer_dropout,
            )
            for _ in range(n_duett_layers)
        ])

        hid = int(np.sqrt(tt_dim))
        self.full_time_embedding = nn.Sequential(
            nn.Linear(1, hid), nn.Tanh(), BatchNormLastDim(hid), nn.Linear(hid, tt_dim)
        )
        self.full_rep_embedding = nn.Embedding(tt_dim, 1)

        d_rep = d_embedding * (d_time_series_num + 1)
        self.head = simple_mlp(
            d_rep, d_target, n_hidden_head, d_hidden_head,
            hidden_batch_norm=True, activation=nn.ReLU
        )
        self.tab_encoder = simple_mlp(
            d_static_num, d_embedding, n_hidden_tab_encoder,
            d_hidden_tab_encoder, hidden_batch_norm=True
        )

    def forward(
        self, xs_static: torch.Tensor, xs_feats: torch.Tensor, xs_times: torch.Tensor
    ) -> torch.Tensor:
        n_vars = xs_feats.shape[2] // 2
        obs_idx = xs_feats[:, :, n_vars : n_vars * 2].long().clamp(0, 15)
        xs_feats[:, :, n_vars : n_vars * 2] = self.n_obs_embedding(obs_idx).squeeze(-1)

        el_in = torch.empty(
            xs_feats.shape[:-1] + (n_vars, 2),
            dtype=xs_feats.dtype,
            device=xs_feats.device,
        )
        el_in[:, :, :, 0] = xs_feats[:, :, :n_vars]
        el_in[:, :, :, 1] = xs_feats[:, :, n_vars : n_vars * 2]

        psi = torch.zeros(
            (xs_feats.shape[0], xs_feats.shape[1] + 1, n_vars + 1, self.d_embedding),
            dtype=xs_feats.dtype,
            device=xs_feats.device,
        )
        for i, el in enumerate(self.embedding_layers):
            psi[:, :-1, i, :] = el(el_in[:, :, i, :])

        psi[:, :-1, -1, :] = self.tab_encoder(xs_static).unsqueeze(1)
        rep_key = self.REPRESENTATION_EMBEDDING_KEY.to(xs_feats.device)
        psi[:, -1, :, :] = self.special_embeddings(rep_key).unsqueeze(0).unsqueeze(1)

        t_embeds = self.full_time_embedding(xs_times.unsqueeze(2))
        w_rep = self.full_rep_embedding.weight.T.unsqueeze(0)
        t_embeds = torch.cat((t_embeds, w_rep.expand(xs_feats.shape[0], -1, -1)), dim=1)

        for et, tt in zip(self.event_transformers, self.time_transformers):
            et_out_shape = (psi.shape[0], psi.shape[2], psi.shape[1], psi.shape[3])
            emb = psi.transpose(1, 2).flatten(2) + self.full_event_embedding.weight
            e_outs = et(emb).view(et_out_shape).transpose(1, 2)
            
            emb = e_outs.flatten(2) + t_embeds
            psi = tt(emb).view(e_outs.shape)

        z_ts = psi.flatten(2)[:, -1, :]
        return self.head(z_ts).squeeze(1)


class DuETT(BaseModel):
    """PyHealth wrapper matching original paper specifications for DuETT.
    
    Args:
        dataset (SampleDataset): The targeted dataset instance.
        n_timesteps (int): Number of time bins used in the task. Defaults to 32.
        d_embedding (int): Hidden embedding scale. Defaults to 24.
        n_duett_layers (int): Layers of crossed transformers. Defaults to 2.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        n_timesteps: int = 32,
        d_embedding: int = 24,
        n_duett_layers: int = 2,
        **kwargs,
    ):
        super().__init__(dataset=dataset)
        self.d_target = self.get_output_size()
        self.label_key = self.label_keys[0]

        if "x_ts" not in self.feature_keys:
            raise ValueError("DuETT requires 'x_ts' in feature_keys")
            
        # Dynamically extract feature count from the fitted processor
        d_time_series_num = dataset.input_processors["x_ts"].n_features
        
        self.core = DuETTCore(
            d_static_num=8,
            d_time_series_num=d_time_series_num,
            d_target=self.d_target,
            d_embedding=d_embedding,
            n_duett_layers=n_duett_layers,
            masked_transform_timesteps=n_timesteps,
            **kwargs,
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        # Safely handle PyHealth 2.0 tuple unpacking
        x_ts = kwargs["x_ts"][0] if isinstance(kwargs["x_ts"], tuple) else kwargs["x_ts"]
        x_static = kwargs["x_static"][0] if isinstance(kwargs["x_static"], tuple) else kwargs["x_static"]
        times = kwargs["times"][0] if isinstance(kwargs["times"], tuple) else kwargs["times"]

        logits = self.core(x_static, x_ts.clone(), times)
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)

        y_prob = self.prepare_y_prob(logits)

        res = {"logit": logits, "y_prob": y_prob}
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key]
            if y_true.ndim == 1:
                y_true = y_true.unsqueeze(1)
            loss = self.get_loss_function()(logits, y_true.float())
            res["loss"] = loss
            res["y_true"] = y_true

        return res