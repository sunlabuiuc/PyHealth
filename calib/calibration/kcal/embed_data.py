import numpy as np
import pandas as pd
import torch
from scipy.stats import multinomial
from torch.utils.data import Dataset


class _IndexSampler:
    def __init__(self, labels, seed=None, group:np.ndarray=None):
        if not isinstance(labels, pd.Series):
            labels = pd.Series(labels, index=np.arange(len(labels)))
        self.labels = labels
        self.index_by_class = {}
        self.class_weights = {}
        for k, tser in self.labels.groupby(self.labels):
            self.index_by_class[k] = tser.index
            self.class_weights[k] = len(tser)
        self.class_weights = pd.Series(self.class_weights) / pd.Series(self.class_weights).sum()
        self.group = group


        self.rs = np.random.RandomState(seed)

    def _sample_class_cnts(self, batch_size):
        
        ret = multinomial.rvs(batch_size, self.class_weights)
        return ret

    def sample(self, batch_size_pred, batch_size):
        weights = 1.
        if self.group is None:
            pred_indices = self.rs.choice(self.labels.index, batch_size_pred, replace=False)
            avoid_supp_indices = pred_indices
        else:
            _pred_group = self.rs.choice(self.group)
            avoid_supp_indices = self.labels.index[self.group == _pred_group]
            pred_indices = self.rs.choice(avoid_supp_indices, batch_size_pred, replace=False)

        # SAMPLE_MODSTRATIFY
        cnt, indices= {}, {}
        weights = []
        for k, all_idx_class_k in self.index_by_class.items():
            all_idx_class_k = all_idx_class_k.difference(avoid_supp_indices)
            indices[k] = _safe_random_choice(all_idx_class_k, batch_size, replace=False, rs=self.rs)
            cnt[k] = (len(indices[k]), len(all_idx_class_k))
            weights.extend([len(all_idx_class_k)/float(len(indices[k]))] * len(indices[k]))
        indices = np.concatenate(list(indices.values()))
        weights = np.asarray(weights)
        pred_indices = pd.Index(pred_indices)
        return pred_indices, indices, cnt, weights

class _EmbedData(Dataset):
    def __init__(self, labels:np.ndarray, embed:np.ndarray, epoch_len=5000, 
                 bs_pred=64, bs_supp=20, group=None, indices=None) -> None:
        self.num_classes = labels.max() + 1

        self.labels, self.indices = labels, indices
        self.embed = embed

        if group is not None:
            group = np.asarray(group)
            if len(set(group)) == 1:
                group = None
        self.group = group
        self.niters_per_epoch = epoch_len
        self.index_sampler = _IndexSampler(self.labels, seed=42, group=self.group)
        self.bs_pred = bs_pred
        self.bs_supp = bs_supp

        self.use_full = epoch_len == 1
        if self.niters_per_epoch == 1 and group is not None:
            self.niters_per_epoch = len(set(group))


    def __len__(self):
        return self.niters_per_epoch

    def __getitem__(self, index):
        if self.use_full:
            if self.group is None or self.niters_per_epoch == 1:
                return {'data': {"supp_embed": torch.tensor(self.embed, dtype=torch.float),
                'supp_target': torch.tensor(self.labels, dtype=torch.long)}}
            unique_groups = sorted(set(self.group))
            full_indices = np.arange(len(self.labels))
            pred_indices = full_indices[self.group == unique_groups[index]]
            indices = full_indices[self.group != unique_groups[index]]
            weights = 1.
        else:
            pred_indices, indices, _, weights = self.index_sampler.sample(self.bs_pred, self.bs_supp)
        data = {
            'weights': torch.tensor(weights, dtype=torch.float) if isinstance(weights, np.ndarray) else weights,
            'pred_embed': torch.tensor(self.embed[pred_indices], dtype=torch.float),
            'supp_embed': torch.tensor(self.embed[indices], dtype=torch.float),
            'supp_target': torch.tensor(self.labels[indices], dtype=torch.long),
            }
        return {'data': data,  'target': torch.tensor(self.labels[pred_indices], dtype=torch.long)}

    @classmethod
    def _collate_func(cls, batch):
        assert len(batch) == 1
        return batch[0]


def _safe_random_choice(a, size, replace=True, p=None, rs=None):
    if size > len(a) and not replace:
        return a
    if rs is None:
        rs = np.random.RandomState()
    return rs.choice(a, size=size, replace=replace, p=p)
    