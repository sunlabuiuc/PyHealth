from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.embedding import EmbeddingModel


class MultimodalBottleneckTransformerEncoder(nn.Module):
    """
    Generalized Bottleneck Transformer Encoder for N modalities.
    Based on "Attention Bottlenecks for Multimodal Fusion" (Nagrani et al., NeurIPS 2021).
    """

    def __init__(
        self,
        n_modality: int,
        bottlenecks_n: int,
        fusion_startidx: int,
        n_layers: int,
        n_head: int,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super(MultimodalBottleneckTransformerEncoder, self).__init__()

        self.n_modality = n_modality
        self.fusion_startidx = fusion_startidx
        self.n_layers = n_layers
        self.n_fusion_layers = n_layers - fusion_startidx
        self.n_prefusion = fusion_startidx
        self.d_model = d_model
        self.n_bottlenecks = bottlenecks_n

        # Shared Bottleneck Tokens
        self.bottlenecks = nn.Parameter(torch.randn(1, bottlenecks_n, d_model))

        # Prefusion Stacks: independent layers per modality
        self.prefusion_stacks = nn.ModuleList([
            nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_head,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    batch_first=True
                ) for _ in range(n_modality)
            ]) for _ in range(self.n_prefusion)
        ])

        # Fusion Stacks: processes [bottleneck_tokens || modality_tokens]
        self.fusion_stacks = nn.ModuleList([
            nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_head,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    batch_first=True
                ) for _ in range(n_modality)
            ]) for _ in range(self.n_fusion_layers)
        ])

    def forward_prefusion(self, enc_inputs: List[torch.Tensor], masks: List[torch.Tensor]) -> List[torch.Tensor]:
        for enc_layers in self.prefusion_stacks:
            enc_outputs = []
            for modal_idx, enc_layer in enumerate(enc_layers):
                # Apply mask to padding tokens (src_key_padding_mask requires True for ignoring)
                # True in mask = invalid/padding
                enc_out = enc_layer(enc_inputs[modal_idx], src_key_padding_mask=~masks[modal_idx] if masks[modal_idx] is not None else None)
                enc_outputs.append(enc_out)
            enc_inputs = enc_outputs
        return enc_inputs

    def forward_fusion(self, enc_inputs: List[torch.Tensor], masks: List[torch.Tensor], bottleneck_tokens: torch.Tensor, valid_modalities: List[torch.Tensor]) -> List[torch.Tensor]:
        # valid_modalities: [B] list of boolean/float tensors indicating if modality is present
        batch_size = enc_inputs[0].size(0)
        
        for modality_encoders in self.fusion_stacks:
            enc_outputs = []
            bottleneck_tokens_modality_sum = torch.zeros_like(bottleneck_tokens)
            sum_of_modalities = torch.zeros(batch_size, 1, 1, device=bottleneck_tokens.device)

            for idx, enc_layer in enumerate(modality_encoders):
                # Concatenate bottleneck tokens with modality tokens
                # bottleneck_tokens: [B, num_bottlenecks, d_model]
                # enc_inputs[idx]: [B, seq_len, d_model]
                fused_input = torch.cat([bottleneck_tokens, enc_inputs[idx]], dim=1)
                
                # Padding mask for bottleneck tokens is always False (i.e. valid)
                # [B, num_bottlenecks] of False
                b_mask = torch.zeros(batch_size, self.n_bottlenecks, dtype=torch.bool, device=fused_input.device)
                
                # Modality padding mask
                m_mask = ~masks[idx] if masks[idx] is not None else torch.zeros(batch_size, enc_inputs[idx].size(1), dtype=torch.bool, device=fused_input.device)
                
                combined_mask = torch.cat([b_mask, m_mask], dim=1)
                
                # Pass through the layer
                enc_out = enc_layer(fused_input, src_key_padding_mask=combined_mask)
                
                # The output consists of processed bottleneck tokens and modality tokens
                # [B, num_bottlenecks, d_model] and [B, seq_len, d_model]
                bottleneck_hidden_tokens = enc_out[:, :self.n_bottlenecks, :]
                modality_hidden_tokens = enc_out[:, self.n_bottlenecks:, :]
                enc_outputs.append(modality_hidden_tokens)
                
                # Average updated bottlenecks from valid modalities
                modality_is_valid = valid_modalities[idx].view(batch_size, 1, 1)
                bottleneck_tokens_modality_sum += bottleneck_hidden_tokens * modality_is_valid
                sum_of_modalities += modality_is_valid

            # Prevent division by zero if all modalities are missing
            # If sum_of_modalities is 0, just pass zeros (or keep previous bottleneck_tokens)
            # sum_of_modalities = torch.clamp(sum_of_modalities, min=1.0)
            avg_divisor = sum_of_modalities.clone()
            avg_divisor[avg_divisor == 0] = 1.0

            bottleneck_tokens = bottleneck_tokens_modality_sum / avg_divisor
            enc_inputs = enc_outputs
            
        return enc_inputs

    def forward(self, enc_inputs: List[torch.Tensor], masks: List[torch.Tensor]) -> List[torch.Tensor]:
        batch_size = enc_inputs[0].size(0)

        # Determine if a modality is valid for each instance in the batch
        # A modality is valid if it has at least one True in its mask
        valid_modalities = []
        for mask, inp in zip(masks, enc_inputs):
            if mask is not None:
                # [B] - True if there's any valid token (1/True)
                valid = mask.any(dim=1).float()
            else:
                valid = torch.ones(batch_size, device=inp.device)
            valid_modalities.append(valid)
            
        bottleneck_tokens = self.bottlenecks.expand(batch_size, -1, -1)
        
        enc_inputs = self.forward_prefusion(enc_inputs, masks)
        enc_inputs = self.forward_fusion(enc_inputs, masks, bottleneck_tokens, valid_modalities)
        
        return enc_inputs


class BottleneckTransformer(BaseModel):
    """Bottleneck Transformer model for PyHealth datasets.

    This model employs a unified multimodal approach by embedding diverse
    feature streams using :class:`EmbeddingModel` and fusing them with
    the Attention Bottleneck mechanism.

    Each modality first prepends a learnable [CLS] token and is processed by
    independent `prefusion` transformer layers. Then, they are processed by
    fusion transformer layers with shared bottleneck tokens. The [CLS] token
    of each modality is extracted, averaged, and fed to the classification head.

    Args:
        dataset (SampleDataset): dataset providing processed inputs.
        embedding_dim (int): shared embedding dimension.
        bottlenecks_n (int): number of shared bottleneck tokens.
        fusion_startidx (int): the layer index at which bottleneck fusion starts.
        num_layers (int): total number of transformer layers (prefusion + fusion).
        heads (int): number of attention heads per transformer block.
        dropout (float): dropout rate applied inside transformer blocks.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {
        ...         "patient_id": "patient-0",
        ...         "visit_id": "visit-0",
        ...         "conditions": ["A", "B", "C"],
        ...         "procedures": ["X", "Y"],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "patient-1",
        ...         "visit_id": "visit-0",
        ...         "conditions": ["D"],
        ...         "procedures": ["Z", "Y"],
        ...         "label": 0,
        ...     },
        ... ]
        >>> input_schema = {"conditions": "sequence", "procedures": "sequence"}
        >>> output_schema = {"label": "binary"}
        >>> dataset = create_sample_dataset(
        ...     samples,
        ...     input_schema,
        ...     output_schema,
        ...     dataset_name="demo",
        ... )
        >>> model = BottleneckTransformer(dataset=dataset, num_layers=3, fusion_startidx=1, bottlenecks_n=4)
        >>> loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>> batch = next(iter(loader))
        >>> output = model(**batch)
        >>> sorted(output.keys())
        ['logit', 'loss', 'y_prob', 'y_true']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        bottlenecks_n: int = 4,
        fusion_startidx: int = 1,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__(dataset=dataset)
        self.embedding_dim = embedding_dim
        self.bottlenecks_n = bottlenecks_n
        self.fusion_startidx = fusion_startidx
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout

        assert (
            len(self.label_keys) == 1
        ), "Only one label key is supported if BottleneckTransformer is initialized"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)
        
        self.n_modality = len(self.feature_keys)
        
        # Classification tokens for each modality
        self.cls_token_per_modality = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, embedding_dim)) for _ in range(self.n_modality)
        ])

        self.encoder = MultimodalBottleneckTransformerEncoder(
            n_modality=self.n_modality,
            bottlenecks_n=bottlenecks_n,
            fusion_startidx=fusion_startidx,
            n_layers=num_layers,
            n_head=heads,
            d_model=embedding_dim,
            d_ff=embedding_dim * 4,
            dropout=dropout
        )

        output_size = self.get_output_size()
        # Outputs of each modality's CLS token are averaged, not concatenated
        self.fc = nn.Linear(embedding_dim, output_size)

    @staticmethod
    def _pool_embedding(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.sum(dim=2)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x

    @staticmethod
    def _mask_from_embeddings(x: torch.Tensor) -> torch.Tensor:
        mask = torch.any(torch.abs(x) > 0, dim=-1)
        if mask.dim() == 1:
            mask = mask.unsqueeze(1)
        invalid_rows = ~mask.any(dim=1)
        if invalid_rows.any():
            mask[invalid_rows, 0] = True
        return mask.bool()

    def forward(
        self,
        **kwargs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **kwargs: keyword arguments for the model.

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the final loss.
                y_prob: a tensor of predicted probabilities.
                y_true: a tensor representing the true labels.
                logit: the raw logits before activation.
        """
        enc_inputs = []
        masks = []
        
        for idx, feature_key in enumerate(self.feature_keys):
            feature = kwargs[feature_key]

            if isinstance(feature, torch.Tensor):
                feature = (feature,)

            schema = self.dataset.input_processors[feature_key].schema()

            value = feature[schema.index("value")] if "value" in schema else None
            mask = feature[schema.index("mask")] if "mask" in schema else None

            if len(feature) == len(schema) + 1 and mask is None:
                mask = feature[-1]

            if value is None:
                raise ValueError(
                    f"Feature '{feature_key}' must contain 'value' "
                    f"in the schema."
                )
            else:
                value = value.to(self.device)
                
            if mask is not None:
                mask = mask.to(self.device)
                value = self.embedding_model({feature_key: value}, masks={feature_key: mask})[feature_key]
            else:
                value = self.embedding_model({feature_key: value})[feature_key]
                
            value = self._pool_embedding(value)
            
            if mask is not None:
                mask = mask.bool()
                if mask.dim() == value.dim():
                    mask = mask.any(dim=-1)
            else:
                mask = self._mask_from_embeddings(value)
            
            # Prepend Modality CLS token
            batch_size = value.size(0)
            cls_token = self.cls_token_per_modality[idx].expand(batch_size, -1, -1)
            value = torch.cat([cls_token, value], dim=1)
            
            # Update mask for CLS token (always valid)
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=value.device)
            mask = torch.cat([cls_mask, mask], dim=1)
            
            enc_inputs.append(value)
            masks.append(mask)

        # Pass through Bottleneck Transformer Encoder
        enc_outputs = self.encoder(enc_inputs, masks)
        
        # Extract CLS tokens
        cls_tokens = [out[:, 0, :].unsqueeze(1) for out in enc_outputs]
        cls_tokens = torch.cat(cls_tokens, dim=1) # [B, n_modality, embedding_dim]
        
        # Average CLS tokens across valid modalities
        b_size = cls_tokens.size(0)
        valid_modalities = []
        for mask in masks:
            # We check if there's any valid token aside from the CLS token (index 0)
            if mask.size(1) > 1:
                valid = mask[:, 1:].any(dim=1).float()
            else:
                valid = mask[:, 0].float() # fallback
            valid_modalities.append(valid.view(b_size, 1, 1))
            
        valid_modality_tensor = torch.cat(valid_modalities, dim=1) # [B, n_modality, 1]
        
        # Apply valid mask
        masked_cls = cls_tokens * valid_modality_tensor
        sum_valid = valid_modality_tensor.sum(dim=1) # [B, 1]
        
        # Avoid division by zero
        sum_valid[sum_valid == 0] = 1.0
        patient_emb = masked_cls.sum(dim=1) / sum_valid # [B, embedding_dim]
        
        logits = self.fc(patient_emb)
        y_prob = self.prepare_y_prob(logits)

        results = {
            "logit": logits,
            "y_prob": y_prob,
        }

        if self.label_key in kwargs:
            y_true = cast(torch.Tensor, kwargs[self.label_key]).to(self.device)
            loss = self.get_loss_function()(logits, y_true)
            results["loss"] = loss
            results["y_true"] = y_true

        return results

if __name__ == "__main__":
    from pyhealth.datasets import create_sample_dataset, get_dataloader

    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "conditions": ["A", "B", "C"],
            "procedures": ["X", "Y"],
            "label": 1,
        },
        {
            "patient_id": "patient-1",
            "visit_id": "visit-0",
            "conditions": ["D"],
            "procedures": ["Z", "Y"],
            "label": 0,
        },
    ]

    input_schema = {
        "conditions": "sequence",
        "procedures": "sequence",
    }
    output_schema = {"label": "binary"}

    dataset = create_sample_dataset(
        samples=samples,
        input_schema=input_schema,
        output_schema=output_schema,
        dataset_name="test",
    )

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    model = BottleneckTransformer(
        dataset=dataset, 
        embedding_dim=64, 
        bottlenecks_n=2, 
        fusion_startidx=1, 
        num_layers=3, 
        heads=2
    )

    data_batch = next(iter(train_loader))

    result = model(**data_batch)
    print(result)

    result["loss"].backward()
    print("Test completed successfully.")
