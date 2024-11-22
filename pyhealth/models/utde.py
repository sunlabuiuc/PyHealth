from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from pyhealth.datasets import SampleEHRDataset
from pyhealth.models import BaseModel, CNNLayer

# VALID_OPERATION_LEVEL = ["visit", "event"]

class multiTimeAttention(nn.Module):
    """MultiTimeAttention module
    borrowed from: https://github.com/XZhang97666/MultimodalMIMIC/blob/main/module.py
    """

    def __init__(self, input_dim, nhidden=16,
                 embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time),
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*num_heads, nhidden)])

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"

        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            if len(mask.shape)==3:
                mask=mask.unsqueeze(-1)

            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -10000)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn=F.dropout(p_attn, p=dropout, training=self.training)
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn


    def forward(self, query, key, value, mask=None, dropout=0.1):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous() \
             .view(batch, -1, self.h * dim)
        return self.linears[-1](x)


class UTDE(nn.Module):
    """Unified Time Discretization-based Embedding module.

    This layer wraps the PyTorch RNN layer with masking and dropout support. It is
    used in the RNN model. But it can also be used as a standalone layer.

    Args:
        input_size: input feature size.
        hidden_size: hidden feature size.
        rnn_type: type of rnn, one of "RNN", "LSTM", "GRU". Default is "GRU".
        num_layers: number of recurrent layers. Default is 1.
        dropout: dropout rate. If non-zero, introduces a Dropout layer before each
            RNN layer. Default is 0.5.
        bidirectional: whether to use bidirectional recurrent layers. If True,
            a fully-connected layer is applied to the concatenation of the forward
            and backward hidden states to reduce the dimension to hidden_size.
            Default is False.

    Examples:
        >>> from pyhealth.models import RNNLayer
        >>> input = torch.randn(3, 128, 5)  # [batch size, sequence len, input_size]
        >>> layer = RNNLayer(5, 64)
        >>> outputs, last_outputs = layer(input)
        >>> outputs.shape
        torch.Size([3, 128, 64])
        >>> last_outputs.shape
        torch.Size([3, 64])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        embed_time: int,
        rnn_type: str = "GRU",
        num_layers: int = 1,
        dropout: float = 0.5,
        bidirectional: bool = False,
        alpha: torch.tensor = None,
    ):
        super(UTDE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_time = embed_time

        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)
        self.time_attn_ts = multiTimeAttention(embed_time*2, embed_time, embed_time, 8)
        self.proj1 = nn.Linear(embed_time, embed_time)
        self.proj2 = nn.Linear(embed_time, embed_time)
        self.out_layer= nn.Linear(embed_time, output_dim)
        self.imputation_conv1d = CNNLayer(5, 64) # TODO: match the dimension with actual immputation
    
    def learn_time_embedding(
        self,
        T: torch.tensor,
    ):
        """Function to compute the time embedding for each features
        
        Args:
            T: a tensor of shape [batch size, sequence len] from discretization step
        
        Returns:
            time_embedding: a tensor of shape [batch size, sequence len, time embedding size],
                time embedding tensor for each type of lab events for mTAND use
        """
        T = T.unsqueeze(-1)
        out1 = self.linear(T)
        out2 = torch.sin(self.periodic(T))
        time_embedding = torch.cat([out1, out2], dim=-1)
        return time_embedding
    
    def imputation(
        self, 
        discretized_feature: torch.tensor,
    ) -> torch.tensor:
        """Getting imputation embedding
        
        Args:
            discretized_feature: a tensor of shape [batch size, input channels, input size] from discretization step
        
        Returns:
            imputation_embedding: a tensor of shape [batch size, output channels, hidden size]
        """
        if len(discretized_feature.shape) == 2:
            discretized_feature.unsqueeze(1)
        return self.imputation_conv1d(discretized_feature)
    
    def mTAND(
        self,
        X: torch.tensor,
        T: torch.tensor,
        alpha: torch.tensor,
    ) -> torch.tensor:
        """Getting discretized multi-time attention embedding
        
        Args:
            X: a tensor of shape [batch size, sequence length],
                original values for different lab events at different time stamps 
            T: a tensor of shape [batch size, sequence length],
                timestamps for the different lab events
        
        Returns:
            imputation_embedding: a tensor of shape [batch size, output channels, hidden size]
        """
        keys = self.learn_time_embedding(T)
        query = self.learn_time_embedding(alpha)
        
        X_irg = torch.cat((X, X_mask), 2)
        X_mask = torch.cat((X_mask, X_mask), 2)
                
        proj_X = self.time_attn_ts(query, keys, X_irg, X_mask).transpose(0, 1)
        
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        return output
    
    def forward(
        self,
        X: torch.tensor,
        T: torch.tensor,
        alpha: torch.tensor,
        discretized_feature: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Forward propagation.

        Args:
            X: a tensor of shape [batch size, sequence length],
                original values for different lab events at different time stamps 
            T: a tensor of shape [batch size, sequence length],
                timestamps for the different lab events
        
        Returns:
            imputation_embedding: a tensor of shape [batch size, output channels, hidden size]
            mtand_embedding: a tensor of shape [batch size, otuput channels, hidden size]
        """
        imputation_embedding = self.imputation(discretized_feature)
        mtand_embedding = self.mTAND(X, T, alpha)
        return imputation_embedding, mtand_embedding

if __name__ == "__main__":
    # from pyhealth.datasets import SampleEHRDataset
    #
    # samples = [
    #     {
    #         "patient_id": "patient-0",
    #         "visit_id": "visit-0",
    #         # "single_vector": [1, 2, 3],
    #         "list_codes": ["505800458", "50580045810", "50580045811"],  # NDC
    #         "list_vectors": [[1.0, 2.55, 3.4], [4.1, 5.5, 6.0]],
    #         "list_list_codes": [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],  # ATC-4
    #         "list_list_vectors": [
    #             [[1.8, 2.25, 3.41], [4.50, 5.9, 6.0]],
    #             [[7.7, 8.5, 9.4]],
    #         ],
    #         "label": 1,
    #     },
    #     {
    #         "patient_id": "patient-0",
    #         "visit_id": "visit-1",
    #         # "single_vector": [1, 5, 8],
    #         "list_codes": [
    #             "55154191800",
    #             "551541928",
    #             "55154192800",
    #             "705182798",
    #             "70518279800",
    #         ],
    #         "list_vectors": [[1.4, 3.2, 3.5], [4.1, 5.9, 1.7], [4.5, 5.9, 1.7]],
    #         "list_list_codes": [["A04A", "B035", "C129"]],
    #         "list_list_vectors": [
    #             [[1.0, 2.8, 3.3], [4.9, 5.0, 6.6], [7.7, 8.4, 1.3], [7.7, 8.4, 1.3]],
    #         ],
    #         "label": 0,
    #     },
    # ]
    #
    # # dataset
    # dataset = SampleEHRDataset(samples=samples, dataset_name="test")
    from pyhealth.datasets import MIMIC4Dataset
    from pyhealth.tasks import Mortality30DaysMIMIC4

    dataset = MIMIC4Dataset(
        root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        tables=["procedures_icd"],
        dev=True,
    )
    task = Mortality30DaysMIMIC4()
    samples = dataset.set_task(task)

    # data loader
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(samples, batch_size=2, shuffle=True)

    # model
    model = RNN(
        dataset=samples,
        feature_keys=[
            "procedures",
        ],
        label_key="mortality",
        mode="binary",
    )

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()
