import torch
import torch.nn as nn
from transformers import AutoModel
from pyhealth.models import BaseModel

# Define a new model that extends PyHealth's BaseModel
class AttentionClinicalBERT(BaseModel):
    def __init__(
        self,
        dataset,
        mode: str = "multilabel",
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        hidden_size: int = 768,
        dropout: float = 0.1,
        num_attention_heads: int = 1,
    ):
        # Initialize the parent BaseModel with dataset and task settings
        super(AttentionClinicalBERT, self).__init__(
            dataset=dataset, mode=mode, feature_keys=["note"], target_keys=["labels"],
        )

        # Load a pretrained ClinicalBERT model
        self.bert = AutoModel.from_pretrained(model_name)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

        # Multihead attention layer applied over token-level BERT outputs
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads, batch_first=True)

        # Final classification layer mapping attention output to target size
        self.classifier = nn.Linear(hidden_size, self.output_size)

    def forward(self, **kwargs):
        # Extract token IDs and attention masks from input
        input_ids = kwargs["note"]["input_ids"]  # shape: [B, T]
        attention_mask = kwargs["note"]["attention_mask"]  # shape: [B, T]

        # Forward pass through BERT to get token-level embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # shape: [B, T, H]

        # Apply self-attention across token embeddings
        # key_padding_mask should be False where tokens are valid
        attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states, key_padding_mask=~attention_mask.bool())

        # Pool the attended output by averaging over the token dimension
        pooled_output = attn_output.mean(dim=1)  # shape: [B, H]

        # Apply dropout to the pooled representation
        pooled_output = self.dropout(pooled_output)

        # Run the pooled output through the classification head
        logits = self.classifier(pooled_output)  # shape: [B, num_labels]

        return logits
