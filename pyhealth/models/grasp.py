import copy
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from sklearn.neighbors import kneighbors_graph

from pyhealth.datasets import SampleEHRDataset
from pyhealth.models import BaseModel, ConCareLayer, RNNLayer
from pyhealth.models.utils import get_last_visit


def random_init(dataset, num_centers, device):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    # print("random size", dataset.size())
    # print("numcenter", num_centers)

    indices = torch.tensor(
        np.array(random.sample(range(num_points), k=num_centers)), dtype=torch.long
    )

    centers = torch.gather(
        dataset, 0, indices.view(-1, 1).expand(-1, dimension).to(device=device)
    )
    return centers


# Compute for each data point the closest center
def compute_codes(dataset, centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    num_centers = centers.size(0)

    # print("size:", dataset.size(), centers.size())
    # 5e8 should vary depending on the free memory on the GPU
    # Ideally, automatically ;)
    chunk_size = int(5e8 / num_centers)
    codes = torch.zeros(num_points, dtype=torch.long)
    centers_t = torch.transpose(centers, 0, 1)
    centers_norms = torch.sum(centers**2, dim=1).view(1, -1)
    for i in range(0, num_points, chunk_size):
        begin = i
        end = min(begin + chunk_size, num_points)
        dataset_piece = dataset[begin:end, :]
        dataset_norms = torch.sum(dataset_piece**2, dim=1).view(-1, 1)
        distances = torch.mm(dataset_piece, centers_t)
        distances *= -2.0
        distances += dataset_norms
        distances += centers_norms
        _, min_ind = torch.min(distances, dim=1)
        codes[begin:end] = min_ind
    return codes


# Compute new centers as means of the data points forming the clusters
def update_centers(dataset, codes, num_centers, device):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    centers = torch.zeros(num_centers, dimension, dtype=torch.float).to(device=device)
    cnt = torch.zeros(num_centers, dtype=torch.float)
    centers.scatter_add_(
        0, codes.view(-1, 1).expand(-1, dimension).to(device=device), dataset
    )
    cnt.scatter_add_(0, codes, torch.ones(num_points, dtype=torch.float))
    # Avoiding division by zero
    # Not necessary if there are no duplicates among the data points
    cnt = torch.where(cnt > 0.5, cnt, torch.ones(num_centers, dtype=torch.float))
    centers /= cnt.view(-1, 1).to(device=device)
    return centers


def cluster(dataset, num_centers, device):
    centers = random_init(dataset, num_centers, device)
    codes = compute_codes(dataset, centers)
    num_iterations = 0
    while True:
        num_iterations += 1
        centers = update_centers(dataset, codes, num_centers, device)
        new_codes = compute_codes(dataset, centers)
        # Waiting until the clustering stops updating altogether
        # This is too strict in practice
        if torch.equal(codes, new_codes):
            break
        if num_iterations > 1000:
            break
        codes = new_codes
    return centers, codes


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features).float())
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features).float())
        else:
            self.register_parameter("bias", None)
        self.initialize_parameters()

    def initialize_parameters(self):
        std = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, adj, x, device):
        y = torch.mm(x.float(), self.weight.float())
        output = torch.mm(adj.float(), y.float())
        if self.bias is not None:
            return output + self.bias.float().to(device=device)
        else:
            return output


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class GRASPLayer(nn.Module):
    """GRASPLayer layer.

    Paper: Liantao Ma et al. GRASP: generic framework for health status representation learning based on incorporating knowledge from similar patients. AAAI 2021.

    This layer is used in the GRASP model. But it can also be used as a
    standalone layer.

    Args:
        input_dim: dynamic feature size.
        static_dim: static feature size, if 0, then no static feature is used.
        hidden_dim: hidden dimension of the GRASP layer, default 128.
        cluster_num: number of clusters, default 12. The cluster_num should be no more than the number of samples.
        dropout: dropout rate, default 0.5.
        block: the backbone model used in the GRASP layer ('ConCare', 'LSTM' or 'GRU'), default 'ConCare'.

    Examples:
        >>> from pyhealth.models import GRASPLayer
        >>> input = torch.randn(3, 128, 64)  # [batch size, sequence len, feature_size]
        >>> layer = GRASPLayer(64, cluster_num=2)
        >>> c = layer(input)
        >>> c.shape
        torch.Size([3, 128])
    """

    def __init__(
        self,
        input_dim: int,
        static_dim: int = 0,
        hidden_dim: int = 128,
        cluster_num: int = 2,
        dropout: int = 0.5,
        block: str = "ConCare",
    ):
        super(GRASPLayer, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cluster_num = cluster_num
        self.dropout = dropout
        self.block = block

        if self.block == "ConCare":
            self.backbone = ConCareLayer(
                input_dim, static_dim, hidden_dim, hidden_dim, dropout=0
            )
        elif self.block == "GRU":
            self.backbone = RNNLayer(input_dim, hidden_dim, rnn_type="GRU", dropout=0)
        elif self.block == "LSTM":
            self.backbone = RNNLayer(input_dim, hidden_dim, rnn_type="LSTM", dropout=0)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.weight1 = nn.Linear(self.hidden_dim, 1)
        self.weight2 = nn.Linear(self.hidden_dim, 1)
        self.GCN = GraphConvolution(self.hidden_dim, self.hidden_dim, bias=True)
        self.GCN.initialize_parameters()
        self.GCN_2 = GraphConvolution(self.hidden_dim, self.hidden_dim, bias=True)
        self.GCN_2.initialize_parameters()
        self.A_mat = None

        self.bn = nn.BatchNorm1d(self.hidden_dim)

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)

        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature, device):
        y = logits + self.sample_gumbel(logits.size()).to(device=device)
        return torch.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, device, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature, device)

        if not hard:
            return y.view(-1, self.cluster_num)

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard

    def grasp_encoder(self, input, static=None, mask=None):

        if self.block == "ConCare":
            hidden_t, _ = self.backbone(input, mask=mask, static=static)
        else:
            _, hidden_t = self.backbone(input, mask)
        hidden_t = torch.squeeze(hidden_t, 0)

        centers, codes = cluster(hidden_t, self.cluster_num, input.device)

        if self.A_mat == None:
            A_mat = np.eye(self.cluster_num)
        else:
            A_mat = kneighbors_graph(
                np.array(centers.detach().cpu().numpy()),
                20,
                mode="connectivity",
                include_self=False,
            ).toarray()

        adj_mat = torch.tensor(A_mat).to(device=input.device)

        e = self.relu(torch.matmul(hidden_t, centers.transpose(0, 1)))  # b clu_num

        scores = self.gumbel_softmax(e, temperature=1, device=input.device, hard=True)
        digits = torch.argmax(scores, dim=-1)  #  b

        h_prime = self.relu(self.GCN(adj_mat, centers, input.device))
        h_prime = self.relu(self.GCN_2(adj_mat, h_prime, input.device))

        clu_appendix = torch.matmul(scores, h_prime)

        weight1 = torch.sigmoid(self.weight1(clu_appendix))
        weight2 = torch.sigmoid(self.weight2(hidden_t))
        weight1 = weight1 / (weight1 + weight2)
        weight2 = 1 - weight1

        final_h = weight1 * clu_appendix + weight2 * hidden_t
        out = final_h
        return out

    def forward(
        self,
        x: torch.tensor,
        static: Optional[torch.tensor] = None,
        mask: Optional[torch.tensor] = None,
    ) -> torch.tensor:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, input_dim].
            static: a tensor of shape [batch size, static_dim].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            output: a tensor of shape [batch size, fusion_dim] representing the
                patient embedding.
        """
        # rnn will only apply dropout between layers
        out = self.grasp_encoder(x, static, mask)
        out = self.dropout(out)
        return out


class GRASP(BaseModel):
    """GRASP model.

    Paper: Liantao Ma et al. GRASP: generic framework for health status representation learning based on incorporating knowledge from similar patients. AAAI 2021.

    Note:
        We use separate GRASP layers for different feature_keys.
        Currently, we automatically support different input formats:
            - code based input (need to use the embedding table later)
            - float/int based value input
        We follow the current convention for the GRASP model:
            - case 1. [code1, code2, code3, ...]
                - we will assume the code follows the order; our model will encode
                each code into a vector and apply GRASP on the code level
            - case 2. [[code1, code2]] or [[code1, code2], [code3, code4, code5], ...]
                - we will assume the inner bracket follows the order; our model first
                use the embedding table to encode each code into a vector and then use
                average/mean pooling to get one vector for one inner bracket; then use
                GRASP one the braket level
            - case 3. [[1.5, 2.0, 0.0]] or [[1.5, 2.0, 0.0], [8, 1.2, 4.5], ...]
                - this case only makes sense when each inner bracket has the same length;
                we assume each dimension has the same meaning; we run GRASP directly
                on the inner bracket level, similar to case 1 after embedding table
            - case 4. [[[1.5, 2.0, 0.0]]] or [[[1.5, 2.0, 0.0], [8, 1.2, 4.5]], ...]
                - this case only makes sense when each inner bracket has the same length;
                we assume each dimension has the same meaning; we run GRASP directly
                on the inner bracket level, similar to case 2 after embedding table

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys:  list of keys in samples to use as features,
            e.g. ["conditions", "procedures"].
        label_key: key in samples to use as label (e.g., "drugs").
        mode: one of "binary", "multiclass", or "multilabel".
        static_keys: the key in samples to use as static features, e.g. "demographics". Default is None.
                     we only support numerical static features.
        use_embedding: list of bools indicating whether to use embedding for each feature type,
            e.g. [True, False].
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension of the GRASP layer. Default is 128.
        cluster_num: the number of clusters. Default is 10. Note that batch size should be greater than cluster_num.
        **kwargs: other parameters for the GRASP layer.


    Examples:
        >>> from pyhealth.datasets import SampleEHRDataset
        >>> samples = [
        ...         {
        ...             "patient_id": "patient-0",
        ...             "visit_id": "visit-0",
        ...             "list_codes": ["505800458", "50580045810", "50580045811"],  # NDC
        ...             "list_vectors": [[1.0, 2.55, 3.4], [4.1, 5.5, 6.0]],
        ...             "list_list_codes": [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],  # ATC-4
        ...             "list_list_vectors": [
        ...                 [[1.8, 2.25, 3.41], [4.50, 5.9, 6.0]],
        ...                 [[7.7, 8.5, 9.4]],
        ...             ],
        ...             "demographic": [0.0, 2.0, 1.5],
        ...             "label": 1,
        ...         },
        ...         {
        ...             "patient_id": "patient-0",
        ...             "visit_id": "visit-1",
        ...             "list_codes": [
        ...                 "55154191800",
        ...                 "551541928",
        ...                 "55154192800",
        ...                 "705182798",
        ...                 "70518279800",
        ...             ],
        ...             "list_vectors": [[1.4, 3.2, 3.5], [4.1, 5.9, 1.7], [4.5, 5.9, 1.7]],
        ...             "list_list_codes": [["A04A", "B035", "C129"]],
        ...             "list_list_vectors": [
        ...                 [[1.0, 2.8, 3.3], [4.9, 5.0, 6.6], [7.7, 8.4, 1.3], [7.7, 8.4, 1.3]],
        ...             ],
        ...             "demographic": [0.0, 2.0, 1.5],
        ...             "label": 0,
        ...         },
        ...     ]
        >>> dataset = SampleEHRDataset(samples=samples, dataset_name="test")
        >>>
        >>> from pyhealth.models import GRASP
        >>> model = GRASP(
        ...         dataset=dataset,
        ...         feature_keys=[
        ...             "list_codes",
        ...             "list_vectors",
        ...             "list_list_codes",
        ...             "list_list_vectors",
        ...         ],
        ...         label_key="label",
        ...         static_key="demographic",
        ...         use_embedding=[True, False, True, False],
        ...         mode="binary"
        ...     )
        >>>
        >>> from pyhealth.datasets import get_dataloader
        >>> train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>> data_batch = next(iter(train_loader))
        >>>
        >>> ret = model(**data_batch)
        >>> print(ret)
        {
            'loss': tensor(0.6896, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>),
            'y_prob': tensor([[0.4983],
                        [0.4947]], grad_fn=<SigmoidBackward0>),
            'y_true': tensor([[1.],
                        [0.]]),
            'logit': tensor([[-0.0070],
                        [-0.0213]], grad_fn=<AddmmBackward0>)
        }
        >>>

    """

    def __init__(
        self,
        dataset: SampleEHRDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        use_embedding: List[bool],
        static_key: Optional[str] = None,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs,
    ):
        super(GRASP, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.use_embedding = use_embedding

        # validate kwargs for GRASP layer
        if "feature_size" in kwargs:
            raise ValueError("feature_size is determined by embedding_dim")
        if len(dataset) < 12 and "cluster_num" not in kwargs:
            raise ValueError("cluster_num is required for small dataset, default 12")
        if "cluster_num" in kwargs and kwargs["cluster_num"] > len(dataset):
            raise ValueError("cluster_num must be no larger than dataset size")

        cluster_num = kwargs.get("cluster_num", 12)

        # the key of self.feat_tokenizers only contains the code based inputs
        self.feat_tokenizers = {}
        self.static_key = static_key
        self.label_tokenizer = self.get_label_tokenizer()
        # the key of self.embeddings only contains the code based inputs
        self.embeddings = nn.ModuleDict()
        # the key of self.linear_layers only contains the float/int based inputs
        self.linear_layers = nn.ModuleDict()

        self.static_dim = 0
        if self.static_key is not None:
            self.static_dim = self.dataset.input_info[self.static_key]["len"]

        self.grasp = nn.ModuleDict()
        # add feature GRASP layers
        for idx, feature_key in enumerate(self.feature_keys):
            input_info = self.dataset.input_info[feature_key]
            # sanity check
            if input_info["type"] not in [str, float, int]:
                raise ValueError(
                    "GRASP only supports str code, float and int as input types"
                )
            elif (input_info["type"] == str) and (input_info["dim"] not in [2, 3]):
                raise ValueError(
                    "GRASP only supports 2-dim or 3-dim str code as input types"
                )
            elif (input_info["type"] == str) and (use_embedding[idx] == False):
                raise ValueError(
                    "GRASP only supports embedding for str code as input types"
                )
            elif (input_info["type"] in [float, int]) and (
                input_info["dim"] not in [2, 3]
            ):
                raise ValueError(
                    "GRASP only supports 2-dim or 3-dim float and int as input types"
                )

            # for code based input, we need Type
            # for float/int based input, we need Type, input_dim
            if use_embedding[idx]:
                self.add_feature_transform_layer(feature_key, input_info)
                self.grasp[feature_key] = GRASPLayer(
                    input_dim=embedding_dim,
                    static_dim=self.static_dim,
                    hidden_dim=self.hidden_dim,
                    **kwargs,
                )
            else:
                self.grasp[feature_key] = GRASPLayer(
                    input_dim=input_info["len"],
                    static_dim=self.static_dim,
                    hidden_dim=self.hidden_dim,
                    **kwargs,
                )

        output_size = self.get_output_size(self.label_tokenizer)
        self.fc = nn.Linear(len(self.feature_keys) * self.hidden_dim, output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        The label `kwargs[self.label_key]` is a list of labels for each patient.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the final loss.
                y_prob: a tensor representing the predicted probabilities.
                y_true: a tensor representing the true labels.
        """
        patient_emb = []
        for idx, feature_key in enumerate(self.feature_keys):
            input_info = self.dataset.input_info[feature_key]
            dim_, type_ = input_info["dim"], input_info["type"]

            # for case 1: [code1, code2, code3, ...]
            if (dim_ == 2) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_2d(
                    kwargs[feature_key]
                )
                # (patient, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                # (patient, event)
                mask = torch.any(x !=0, dim=2)

            # for case 2: [[code1, code2], [code3, ...], ...]
            elif (dim_ == 3) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    kwargs[feature_key]
                )
                # (patient, visit, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, visit, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                # (patient, visit, embedding_dim)
                x = torch.sum(x, dim=2)
                # (patient, visit)
                mask = torch.any(x !=0, dim=2)

            # for case 3: [[1.5, 2.0, 0.0], ...]
            elif (dim_ == 2) and (type_ in [float, int]):
                x, mask = self.padding2d(kwargs[feature_key])
                # (patient, event, values)
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                # (patient, event, embedding_dim)
                if self.use_embedding[idx]:
                    x = self.linear_layers[feature_key](x)
                # (patient, event)
                mask = mask.bool().to(self.device)

            # for case 4: [[[1.5, 2.0, 0.0], [1.8, 2.4, 6.0]], ...]
            elif (dim_ == 3) and (type_ in [float, int]):
                x, mask = self.padding3d(kwargs[feature_key])
                # (patient, visit, event, values)
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                # (patient, visit, embedding_dim)
                x = torch.sum(x, dim=2)

                if self.use_embedding[idx]:
                    x = self.linear_layers[feature_key](x)
                # (patient, event)
                mask = mask[:, :, 0]
                mask = mask.bool().to(self.device)
            else:
                raise NotImplementedError

            if self.static_dim > 0:
                static = torch.tensor(
                    kwargs[self.static_key], dtype=torch.float, device=self.device
                )
                x = self.grasp[feature_key](x, static=static, mask=mask)
            else:
                x = self.grasp[feature_key](x, mask=mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results


if __name__ == "__main__":
    from pyhealth.datasets import SampleEHRDataset

    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            # "single_vector": [1, 2, 3],
            "list_codes": ["505800458", "50580045810", "50580045811"],  # NDC
            "list_vectors": [[1.0, 2.55, 3.4], [4.1, 5.5, 6.0]],
            "list_list_codes": [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],  # ATC-4
            "list_list_vectors": [
                [[1.8, 2.25, 3.41], [4.50, 5.9, 6.0]],
                [[7.7, 8.5, 9.4]],
            ],
            "label": 1,
            "demographic": [1.0, 2.0, 1.3],
        },
        {
            "patient_id": "patient-0",
            "visit_id": "visit-1",
            # "single_vector": [1, 5, 8],
            "list_codes": [
                "55154191800",
                "551541928",
                "55154192800",
                "705182798",
                "70518279800",
            ],
            "list_vectors": [[1.4, 3.2, 3.5], [4.1, 5.9, 1.7], [4.5, 5.9, 1.7]],
            "list_list_codes": [["A04A", "B035", "C129"]],
            "list_list_vectors": [
                [[1.0, 2.8, 3.3], [4.9, 5.0, 6.6], [7.7, 8.4, 1.3], [7.7, 8.4, 1.3]],
            ],
            "label": 0,
            "demographic": [1.0, 2.0, 1.3],
        },
    ]

    # dataset
    dataset = SampleEHRDataset(samples=samples, dataset_name="test")

    # data loader
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    # model
    model = GRASP(
        dataset=dataset,
        feature_keys=[
            "list_codes",
            "list_vectors",
            "list_list_codes",
            # "list_list_vectors",
        ],
        static_key="demographic",
        label_key="label",
        use_embedding=[True, False, True],
        mode="binary",
        cluster_num=2,
    )

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()
