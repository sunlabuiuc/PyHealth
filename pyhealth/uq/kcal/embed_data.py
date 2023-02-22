import ipdb
import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset


class _IndexSampler:
    def __init__(self, labels, seed=None):
        if not isinstance(labels, pd.Series):
            labels = pd.Series(labels, index=np.arange(len(labels)))
        self.labels = labels
        self.index_by_class = {}
        self.class_weights = {}
        for k, tser in self.labels.groupby(self.labels):
            self.index_by_class[k] = tser.index
            self.class_weights[k] = len(tser)
        self.class_weights = pd.Series(self.class_weights) / pd.Series(self.class_weights).sum()

        self.rs = np.random.RandomState(seed)

    def _sample_class_cnts(self, batch_size):
        from scipy.stats import multinomial
        ret = multinomial.rvs(batch_size, self.class_weights)
        return ret

    def sample(self, batch_size_pred, batch_size):
        weights = 1.
        pred_indices = {}
        for k, cnt in enumerate(self._sample_class_cnts(batch_size_pred)):
            pred_indices[k] = self.rs.choice(self.index_by_class[k], cnt, replace=False)
        
        # SAMPLE_MODSTRATIFY
        cnt, indices= {}, {}
        weights = []
        for k, all_idx_class_k in self.index_by_class.items():
            all_idx_class_k = all_idx_class_k.difference(pred_indices[k])
            indices[k] = safe_random_choice(all_idx_class_k, batch_size, replace=False, rs=self.rs)
            cnt[k] = (len(indices[k]), len(all_idx_class_k))
            weights.extend([len(all_idx_class_k)/float(len(indices[k]))] * len(indices[k]))
        indices = np.concatenate(list(indices.values()))
        weights = np.asarray(weights)
        pred_indices = pd.Index(np.concatenate(list(pred_indices.values())))

        return pred_indices, indices, cnt, weights

class _EmbedData(Dataset):
    def __init__(self, labels:pd.Series, embed, niters_per_epoch=5000, bs_pred=64, bs_supp=20) -> None:
        self.num_classes = labels.max() + 1

        self.labels, self.indices = labels.values, labels.index
        self.embed = embed
        self.niters_per_epoch = niters_per_epoch
        self.index_sampler = _IndexSampler(self.labels, seed=42)
        self.bs_pred = bs_pred
        self.bs_supp = bs_supp

    def __len__(self):
        return self.niters_per_epoch

    def __getitem__(self, index):
        if self.niters_per_epoch == 1:
            return {'data': {"supp_embed": torch.tensor(self.embed, dtype=torch.float),
                'supp_target': torch.tensor(self.labels, dtype=torch.long)}}
        pred_indices, indices, _, weights = self.index_sampler.sample(self.bs_pred, self.bs_supp)
        data = {'weights': torch.tensor(weights, dtype=torch.float),
                'pred_embed': torch.tensor(self.embed[pred_indices], dtype=torch.float),
                'supp_embed': torch.tensor(self.embed[indices], dtype=torch.float),
                'supp_target': torch.tensor(self.labels[indices], dtype=torch.long),
                }
        return {'data': data,  'target': torch.tensor(self.labels[pred_indices], dtype=torch.long)}

    @classmethod
    def _collate_func(cls, batch):
        assert len(batch) == 1
        return batch[0]


def safe_random_choice(a, size, replace=True, p=None, rs=None):
    if size > len(a) and not replace:
        return a
    if rs is None:
        rs = np.random.RandomState()
    return rs.choice(a, size=size, replace=replace, p=p)
    