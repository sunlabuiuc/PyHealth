import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import torch.nn as nn


class DeepNote(nn.Module):
    pass



def load_graph(datasets, device, threshold=0.995, return_statics=False):
    train_set, val_set, test_set = datasets
    node_feat = []
    labels = []
    id_map = {}

    def read_data(dataset):
        for i in range(len(dataset)):
            pid = dataset[i]['patient_id']
            feat = dataset[i]['embedding']
            label = dataset[i]['label']
            if pid not in id_map:
                id_map[pid] = len(node_feat)
                node_feat.append(feat)
                labels.append(label)
            else:
                node_feat[id_map[pid]].append(feat)

    read_data(train_set)
    num_train = len(id_map)
    read_data(val_set)
    num_val = len(id_map) - num_train
    read_data(test_set)
    num_test = len(id_map) - num_train - num_val

    node_feat = torch.stack([
        torch.stack(feat_list, dim=0).mean(0) for feat_list in node_feat
    ], dim=0).to(device)
    labels = torch.LongTensor(labels).to(device)
    nids = torch.arange(len(id_map)).to(device)
    train_mask = nids < num_train
    val_mask = (nids >= num_train) & (nids < num_train + num_val)
    test_mask = nids >= num_train + num_val
    node_unit = F.normalize(node_feat, dim=-1)
    cos_sim = node_unit @ node_unit.t()
    edge_index = (cos_sim >= threshold).nonzero().t()
    g = Data(x=node_feat, edge_index=edge_index, train_mask=train_mask,
             val_mask=val_mask, test_mask=test_mask, label=labels)
    if return_statics:
        sparsity = g.num_edges / (g.num_nodes ** 2)
        statics = {'threshold': threshold, 'num_nodes': g.num_nodes, 'num_edges': g.num_edges,
                   'sparsity': sparsity, 'num_train': num_train, 'num_val': num_val, 'num_test': num_test}
        return g, statics
    else:
        print(f'Graph built with {g.num_nodes} nodes and {g.num_edges} edges')
        print(f'Training set: {num_train}')
        print(f'Valid set: {num_val}')
        print(f'Testing set: {num_test}')
    return g