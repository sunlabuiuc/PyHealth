from pyhealth.models import BaseModel
from pyhealth.datasets.sample_dataset import SampleDataset
import torch
import torch.nn as nn

class CADRETransformer(BaseModel):
    def __init__(self, dataset: SampleDataset, gene_emb_matrix, num_drugs,
                 emb_dim=200, num_heads=8, num_layers=4,
                 dropout=0.2, pooling='mean'):
        """Initializes the CADRETransformer model

        Args:
            dataset (SampleDataset): The dataset object containing information about the samples
            gene_emb_matrix (torch.Tensor): Pretrained gene embedding matrix of shape (num_genes, emb_dim)
            num_drugs (int): The total number of unique drugs in the dataset
            emb_dim (int, optional): The dimensionality of the embeddings and defaults to 200
            num_heads (int, optional): The number of attention heads in the Transformer encoder and defaults to 8
            num_layers (int, optional): The number of layers in the Transformer encoder and defaults to 4
            dropout (float, optional): The dropout probability and defaults to 0.2
            pooling (str, optional): The pooling strategy to apply over the encoded gene embeddings ('mean' or 'max') and defaults to 'mean'
        """

        super().__init__(dataset)

        assert pooling in ['mean', 'max'], "pooling must be 'mean' or 'max'"
        self.pooling = pooling

        # Frozen pretrained gene embeddings
        self.gene_emb = nn.Embedding.from_pretrained(gene_emb_matrix, freeze=True)

        # Transformer encoder setup
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            batch_first=True # (B, T, D)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learnable drug embeddings
        self.drug_emb = nn.Embedding(num_drugs, emb_dim)

        self.final_dropout = nn.Dropout(dropout)

    def forward(self, sample: dict) -> torch.Tensor:
        """Forward pass of the CADRETransformer model

        Args:
            sample (dict): A dictionary containing the input data for the model
                           It should contain 'gene_indices' (torch.Tensor of shape (B, L))
                           and 'drug_ids' (torch.Tensor of shape (B,))

        Returns:
            torch.Tensor: Tensor of predicted drug response probabilities of shape (B,)
        """

        gene_indices = sample['gene_indices']
        drug_ids = sample['drug_ids']

        gene_embed = self.gene_emb(gene_indices) # [B, L, D]
        encoded = self.encoder(gene_embed) # [B, L, D]

        # Pooling to fixed-size cell line embedding
        if self.pooling == 'mean':
            cell_embed = encoded.mean(dim=1) # [B, D]
        elif self.pooling == 'max':
            cell_embed = encoded.max(dim=1).values # [B, D]

        cell_embed = self.final_dropout(cell_embed)

        drug_embed = self.drug_emb(drug_ids) # [B, D]

        # Dot product prediction
        dot = torch.sum(cell_embed * drug_embed, dim=1) # [B]
        return torch.sigmoid(dot)