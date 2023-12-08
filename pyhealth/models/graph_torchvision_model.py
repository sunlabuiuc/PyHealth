"""
Recommended:
 - torch: 1.9.1
 - torchvision: 0.10.0+cu102
 - torch_sparse: 0.6.12
"""

from typing import List, Dict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import sys
sys.path.append('.')

from pyhealth.datasets.sample_dataset_v2 import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.sampler import NeighborSampler

SUPPORTED_MODELS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
    "vit_h_14",
    "swin_t",
    "swin_s",
    "swin_b",
]

SUPPORTED_MODELS_FINAL_LAYER = {}
for model in SUPPORTED_MODELS:
    if "resnet" in model:
        SUPPORTED_MODELS_FINAL_LAYER[model] = "fc"
    elif "densenet" in model:
        SUPPORTED_MODELS_FINAL_LAYER[model] = "classifier"
    elif "vit" in model:
        SUPPORTED_MODELS_FINAL_LAYER[model] = "heads.head"
    elif "swin" in model:
        SUPPORTED_MODELS_FINAL_LAYER[model] = "head"
    else:
        raise NotImplementedError


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, init='xavier'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        if init == 'uniform':
            print("| Uniform Initialization")
            self.reset_parameters_uniform()
        elif init == 'xavier':
            print("| Xavier Initialization")
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            print("| Kaiming Initialization")
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        # print("adj", adj.dtype, "support", support.dtype)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, init):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, init=init)
        self.gc2 = GraphConvolution(nhid, nclass, init=init)
        self.dropout = dropout

    def bottleneck(self, path1, path2, path3, adj, in_x):
        return F.relu(path3(F.relu(path2(F.relu(path1(in_x, adj)), adj)), adj))

    def to_sparse_adj(self, adj, size):
        return torch.sparse_coo_tensor(adj.edge_index, torch.ones_like(adj.edge_index[0]), size = size, dtype = torch.float32)

    def forward(self, x, adjs):
        
        temp = self.to_sparse_adj(adjs[0], size = (adjs[0].size[0], adjs[0].size[0]))
        x = F.dropout(F.relu(self.gc1(x, temp)), self.dropout, training=self.training)
        temp = self.to_sparse_adj(adjs[1], size = (adjs[0].size[0], adjs[0].size[0]))
        x = self.gc2(x, temp)

        return F.log_softmax(x, dim=1)


class Graph_TorchvisionModel(BaseModel):
    """Models from PyTorch's torchvision package.

    This class is a wrapper for models from torchvision. It will automatically load
    the corresponding model and weights from torchvision. The final layer will be
    replaced with a linear layer with the correct output size.

    -----------------------------------ResNet------------------------------------------
    Paper: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning
    for Image Recognition. CVPR 2016.
    -----------------------------------DenseNet----------------------------------------
    Paper: Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.
    Densely Connected Convolutional Networks. CVPR 2017.
    ----------------------------Vision Transformer (ViT)-------------------------------
    Paper: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, 
    Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer,
    Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. An Image is Worth
    16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021.
    ----------------------------Swin Transformer (and V2)------------------------------
    Paper: Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, 
    Baining Guo. Swin Transformer: Hierarchical Vision Transformer Using Shifted
    Windows. ICCV 2021.

    Paper: Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, 
    Yue Cao, Zheng Zhang, Li Dong, Furu Wei, Baining Guo. Swin Transformer V2: Scaling
    Up Capacity and Resolution. CVPR 2022.
    -----------------------------------------------------------------------------------
    ----------------------- Graph Convolutional Networks (GCN)-------------------------
    Paper: Thomas N. Kipf, Max Welling. 
    Semi-Supervised Classification with Graph Convolutional Networks. ICLR 2017.
    -----------------------------------------------------------------------------------

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys:  list of keys in samples to use as features, e.g., ["image"].
            Only one feature is supported.
        label_key: key in samples to use as label, e.g., "drugs".
        mode: one of "binary", "multiclass", or "multilabel".
        model_name: str, name of the model to use, e.g., "resnet18".
            See SUPPORTED_MODELS in the source code for the full list.
        model_config: dict, kwargs to pass to the model constructor,
            e.g., {"weights": "DEFAULT"}. See the torchvision documentation for the
            set of supported kwargs for each model.
    -----------------------------------------------------------------------------------
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        model_name: str,
        model_config: dict,
        gnn_config: dict,
    ):
        super(Graph_TorchvisionModel, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )

        self.model_name = model_name
        self.model_config = model_config
        self.gnn_config = gnn_config

        assert len(feature_keys) == 1, "Only one feature is supported!"
        assert model_name in SUPPORTED_MODELS_FINAL_LAYER.keys(), \
            f"PyHealth does not currently include {model_name} model!"

        # for torchvision 0.10.0
        self.model = torchvision.models.__dict__[model_name](**model_config)
        final_layer_name = SUPPORTED_MODELS_FINAL_LAYER[model_name]
        final_layer = self.model
        for name in final_layer_name.split("."):
            final_layer = getattr(final_layer, name)
        hidden_dim = final_layer.in_features
        
        # Graph Model Configs
        gnn_input_dim = gnn_config["input_dim"]
        gnn_hidden_dim = gnn_config["hidden_dim"]

        self.label_tokenizer = self.get_label_tokenizer()
        output_size = self.get_output_size(self.label_tokenizer)
        self.gnn = GCN(nfeat=gnn_input_dim, nhid=gnn_hidden_dim, nclass=output_size, dropout=0.5, init='uniform')
        
        setattr(self.model, final_layer_name.split(".")[0], nn.Linear(hidden_dim, gnn_input_dim))


    def build_graph(self, data, random = False) -> Dict[str, torch.Tensor]:
        """This module generate edge index of graph structure based on given data.
        Currently, we do not have multi-modal data, so this module randomly generate edge index"""
        
        if random:
            edge_index = torch.randint(len(data), size = (2, int(1.0 * len(data))))

        return {
            'edge_index': edge_index
        }


    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation."""
        # concat the info within one batch (batch, channel, length)
        x = kwargs["image"]
        x = torch.stack(x, dim=0).to(self.device)
        if x.shape[1] == 1:
            x = x.repeat((1, 3, 1, 1))
        img_embs = self.model(x)
        logits = self.gnn(img_embs, kwargs["adjacencies"])
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
        }


if __name__ == "__main__":
    
    from torchvision import transforms
    from pyhealth.datasets import COVID19CXRDataset
    from pyhealth.datasets import split_by_sample

    base_dataset = COVID19CXRDataset(
        root="/srv/local/data/COVID-19_Radiography_Dataset",
    )

    sample_dataset = base_dataset.set_task()

    transform = transforms.Compose([
        # transforms.Grayscale(),
        transforms.Lambda(lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1)),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5862785803043838], std=[0.27950088968644304])
    ])


    def encode(sample):
        sample["path"] = transform(sample["path"])
        return sample

    sample_dataset.set_transform(encode)
    
    
    # Get Index of train, valid, test set
    train_index, val_index, test_index = split_by_sample(
        dataset=sample_dataset,
        ratios=[0.7, 0.1, 0.2],
        get_index = True
    )

    model = Graph_TorchvisionModel(
        dataset=sample_dataset,
        feature_keys=["path"],
        label_key="label",
        mode="multiclass",
        model_name="resnet18",
        # model_config={"weights": "DEFAULT"},
        model_config={},
        gnn_config={"input_dim": 256, "hidden_dim": 128},
    )

    graph = model.build_graph(sample_dataset, random = True)
    
    from pyhealth.datasets import split_by_sample

    # Get Index of train, valid, test set
    train_index, val_index, test_index = split_by_sample(
        dataset=sample_dataset,
        ratios=[0.7, 0.1, 0.2],
        get_index = True
    )
    
    # Define Sampler as Dataloader
    train_dataloader = NeighborSampler(sample_dataset, graph["edge_index"], node_idx=train_index, sizes=[15, 10], batch_size=64, shuffle=True, num_workers=12)
    # We sample all edges connected to target node for validation and test (Sizes = [-1, -1])
    valid_dataloader = NeighborSampler(sample_dataset, graph["edge_index"], node_idx=val_index, sizes=[-1, -1], batch_size=64, shuffle=False, num_workers=12)
    test_dataloader = NeighborSampler(sample_dataset, graph["edge_index"], node_idx=test_index, sizes=[-1, -1], batch_size=64, shuffle=False, num_workers=12)

    
    # train_dataloader = NeighborSampler(sample_dataset, graph["edge_index"], node_idx=train_index, sizes=[15, 10], batch_size=64, shuffle=True, num_workers=12)

    # data_graph_batch = next(iter(train_dataloader))

    # # try the model
    # ret = model(**data_graph_batch)
    # print(ret)

    # # try loss backward
    # ret["loss"].backward()
    
    from pyhealth.trainer import Trainer
    resnet_trainer = Trainer(model=model, device="cpu")
    resnet_trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=valid_dataloader,
        epochs=1,
        monitor="accuracy",
    )
    
    print(resnet_trainer.evaluate(test_dataloader))
    
    
    resnet_trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=valid_dataloader,
        epochs=1,
        monitor="accuracy",
    )
