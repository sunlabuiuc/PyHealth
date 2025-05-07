"""
Nikhil Sarin (nsarin2)
DL4H Final Project Contribution

AttentionNet: A Simple Attention-based Model for Healthcare Predictive Tasks

A simple attention-based neural network model for healthcare predictive tasks.
The model uses a self-attention mechanism to focus on important aspects of 
patient health data for various prediction tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..datasets import SampleDataset


class SelfAttentionLayer(nn.Module):
    """Implements a self-attention mechanism for healthcare data.
    
    This layer applies self-attention to input sequences, allowing the model
    to focus on important features in patient health data.
    
    Attributes:
        input_dim: The dimension of input features.
        hidden_dim: The dimension of hidden representations.
        dropout: Dropout rate.
    """
    
    def __init__(self, input_dim, hidden_dim, dropout=0.5):
        """Initializes the SelfAttentionLayer.
        
        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (int): Dimension of hidden representations.
            dropout (float): Dropout rate, default is 0.5.
            
        Returns:
            None
        """
        super(SelfAttentionLayer, self).__init__()
        
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))
        
    def forward(self, x, mask=None):
        """Forward pass for the self-attention layer.
        
        Will be called in model forward method.
        
        Args:
            x (torch.Tensor): Input sequence of shape [batch_size, seq_len, input_dim].
            mask (torch.Tensor, optional): Mask tensor for padded sequences.
            
        Returns:
            torch.Tensor: Contextualized representation after self-attention.
        """
        # Get sequence length
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.query(x)  # [batch_size, seq_len, hidden_dim]
        K = self.key(x)    # [batch_size, seq_len, hidden_dim]
        V = self.value(x)  # [batch_size, seq_len, hidden_dim]
        
        # Scale dot-product attention
        # [batch_size, seq_len, seq_len]
        scale = self.scale.to(x.device)
        attention = torch.matmul(Q, K.transpose(-2, -1)) / scale
        
        # Apply mask if provided
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        
        # Apply softmax and dropout
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        # Get output
        # [batch_size, seq_len, hidden_dim]
        output = torch.matmul(attention, V)
        
        return output


class AttentionNet(nn.Module):
    """AttentionNet model for healthcare predictive tasks.
    
    A simple attention-based neural network model for healthcare predictive tasks.
    The model uses a self-attention mechanism to focus on important aspects of 
    patient health data for various prediction tasks.
    
    Attributes:
        dataset (SampleDataset): The dataset object providing patient data samples.
        feature_keys (list): List of feature keys to use from patient data.
        label_keys (list): List of target label keys.
        embedding_dim (int): The dimension of embeddings.
        hidden_dim (int): The dimension of hidden representations.
        attention_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
    """
    
    def __init__(
        self,
        dataset: SampleDataset,
        feature_keys=None,
        label_keys=None,
        embedding_dim=128,
        hidden_dim=256,
        attention_heads=4,
        dropout=0.5,
    ):
        """Initializes the AttentionNet model.
        
        Args:
            dataset (SampleDataset): The dataset object providing patient data samples.
            feature_keys (list, optional): List of feature keys to use from patient data.
                If None, uses all keys in dataset.input_schema.
            label_keys (list, optional): List of target label keys.
                If None, uses all keys in dataset.output_schema.
            embedding_dim (int, optional): The dimension of embeddings, default is 128.
            hidden_dim (int, optional): The dimension of hidden representations, default is 256.
            attention_heads (int, optional): Number of attention heads, default is 4.
            dropout (float, optional): Dropout rate, default is 0.5.
            
        Returns:
            None
        """
        super(AttentionNet, self).__init__()
        
        self.dataset = dataset
        self.feature_keys = feature_keys if feature_keys is not None else list(dataset.input_schema.keys())
        self.label_keys = label_keys if label_keys is not None else list(dataset.output_schema.keys())
        
        # Parameters
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.attention_heads = attention_heads
        self.dropout_rate = dropout
        
        # Used to query the device of the model
        self._dummy_param = nn.Parameter(torch.empty(0))
        
        # Initialize feature embeddings
        self.embeddings = nn.ModuleDict()
        for key in self.feature_keys:
            # Get vocabulary size for the feature
            vocab_size = self.dataset.input_processors[key].size()
            if vocab_size > 0:
                self.embeddings[key] = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Initialize attention layers
        self.attention_layers = nn.ModuleList(
            [SelfAttentionLayer(embedding_dim, hidden_dim, dropout) for _ in range(attention_heads)]
        )
        
        # Initialize output layers for each label key
        self.output_layers = nn.ModuleDict()
        for key in self.label_keys:
            mode = dataset.output_schema[key]
            if mode == "binary":
                output_dim = 1
            elif mode in ["multiclass", "multilabel"]:
                output_dim = self.dataset.output_processors[key].size()
            elif mode == "regression":
                output_dim = 1
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            # Shared feature layer
            self.output_layers[key] = nn.Sequential(
                nn.Linear(hidden_dim * attention_heads, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
    
    @property
    def device(self) -> torch.device:
        """
        Gets the device of the model.

        Returns:
            torch.device: The device on which the model is located.
        """
        return self._dummy_param.device
        
    def forward(self, **kwargs):
        """Forward pass for the AttentionNet model.
        
        Args:
            **kwargs: Keyword arguments containing feature data.
            
        Returns:
            dict: A dictionary mapping label keys to predicted logits.
        """
        # Get batch size
        batch_size = kwargs[self.feature_keys[0]].size(0)
        
        # Process each feature
        feature_outputs = []
        for key in self.feature_keys:
            x = kwargs[key]
            if key in self.embeddings:
                x = self.embeddings[key](x)
                
                # Apply attention layers
                attention_outputs = []
                for attention_layer in self.attention_layers:
                    # Apply attention
                    attention_output = attention_layer(x)
                    # Mean pooling over sequence dimension
                    attention_output = self.mean_pooling(attention_output)
                    attention_outputs.append(attention_output)
                
                # Concatenate outputs from all attention heads
                feature_output = torch.cat(attention_outputs, dim=1)
                feature_outputs.append(feature_output)
        
        # Combine all feature outputs
        if feature_outputs:
            combined = torch.mean(torch.stack(feature_outputs), dim=0)
        else:
            # If no features have embeddings, return zeros
            combined = torch.zeros(batch_size, self.hidden_dim * self.attention_heads).to(self.device)
        
        # Apply output layers for each label key
        outputs = {}
        for key in self.label_keys:
            outputs[key] = self.output_layers[key](combined)
        
        return outputs
    
    def predict(self, **kwargs):
        """Predicts the output probabilities.
        
        This function performs a forward pass and applies the appropriate activation
        function based on the task mode.
        
        Args:
            **kwargs: Keyword arguments containing feature data.
            
        Returns:
            dict: A dictionary mapping label keys to predicted probabilities.
        """
        logits = self.forward(**kwargs)
        probs = {}
        
        for key in self.label_keys:
            mode = self.dataset.output_schema[key]
            if mode == "binary":
                probs[key] = torch.sigmoid(logits[key])
            elif mode == "multiclass":
                probs[key] = F.softmax(logits[key], dim=1)
            elif mode == "multilabel":
                probs[key] = torch.sigmoid(logits[key])
            elif mode == "regression":
                probs[key] = logits[key]
            else:
                raise ValueError(f"Unknown mode: {mode}")
        
        return probs
    
    def get_loss_function(self, key):
        """Gets the appropriate loss function for a given label key.
        
        Args:
            key (str): The label key.
            
        Returns:
            callable: The loss function.
        """
        mode = self.dataset.output_schema[key]
        if mode == "binary":
            return F.binary_cross_entropy_with_logits
        elif mode == "multiclass":
            return F.cross_entropy
        elif mode == "multilabel":
            return F.binary_cross_entropy_with_logits
        elif mode == "regression":
            return F.mse_loss
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def mean_pooling(self, x, mask=None):
        """Mean pooling operation.
        
        Helper function to perform mean pooling over sequence dimension.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_dim].
            mask (torch.Tensor, optional): Mask tensor for padded sequences.
            
        Returns:
            torch.Tensor: Tensor after mean pooling.
        """
        if mask is not None:
            # Apply mask
            x = x * mask.unsqueeze(-1)
            # Sum and divide by number of non-zero elements
            output = x.sum(dim=1) / mask.sum(dim=1, keepdim=True)
            return output
        else:
            # Simple mean pooling
            return x.mean(dim=1)
    
    def sum_pooling(self, x, mask=None):
        """Sum pooling operation.
        
        Helper function to perform sum pooling over sequence dimension.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_dim].
            mask (torch.Tensor, optional): Mask tensor for padded sequences.
            
        Returns:
            torch.Tensor: Tensor after sum pooling.
        """
        if mask is not None:
            # Apply mask
            x = x * mask.unsqueeze(-1)
        # Sum along sequence dimension
        return x.sum(dim=1)


# Example usage and test case
def main():
    """Example usage for AttentionNet model.
    
    This function shows how to use the AttentionNet model for a simple healthcare task.
    
    Args:
        None
    
    Returns:
        None
    """
    # Import necessary modules
    from pyhealth.datasets import MIMIC3Dataset
    from pyhealth.tasks import readmission_prediction_mimic3_fn
    from pyhealth.datasets import split_by_patient, get_dataloader
    from pyhealth.trainer import Trainer
    
    # Load MIMIC-III dataset
    mimic3base = MIMIC3Dataset(
        root="your_mimic3_path",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD"],
        code_mapping={"ICD9CM": "CCS"},
    )
    
    # Set task
    mimic3sample = mimic3base.set_task(task_fn=readmission_prediction_mimic3_fn)
    
    # Split dataset
    train_ds, val_ds, test_ds = split_by_patient(mimic3sample, [0.8, 0.1, 0.1])
    
    # Create dataloaders
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)
    
    # Initialize AttentionNet model
    model = AttentionNet(
        dataset=mimic3sample,
        feature_keys=["conditions", "procedures"],
        label_keys=["label"],
        embedding_dim=128,
        hidden_dim=256,
        attention_heads=4,
        dropout=0.5,
    )
    
    # Initialize trainer
    trainer = Trainer(model=model)
    
    # Train model
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=50,
        monitor="roc_auc",
    )
    
    # Evaluate model
    metrics = trainer.evaluate(test_loader)
    print(metrics)


if __name__ == "__main__":
    main()