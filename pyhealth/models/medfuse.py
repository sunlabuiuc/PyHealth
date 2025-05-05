import torch
import torch.nn as nn
import torchvision
from torchvision import models # Explicit import for clarity
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pyhealth.models import BaseModel
from pyhealth.datasets import SampleDataset
from typing import Dict, List, Optional, Any

# --- Explanatory Comment Block ---
# NOTE TO REVIEWERS / FUTURE USERS:
#
# This MedFuse model implementation is adapted for the PyHealth library based on
# the architecture described in the original MedFuse paper. It specifically
# implements the LSTM-based fusion mechanism.
#
# **Current Status & Dependencies:**
# This model class defines the neural network architecture and conforms to the
# PyHealth `BaseModel` interface. However, it relies on several assumptions
# about the input `SampleDataset` and the upstream data processing (Task)
# which are *not* yet implemented in the standard PyHealth library.
#
# To make this model fully runnable, the following components would need to be
# created:
#
# 1.  **Custom `MedFuseTask`:** A new task class inheriting from `pyhealth.tasks.BaseTask`
#     is required. This task would be responsible for:
#     * Processing patient data (likely from `MIMIC4Dataset` combining EHR and CXR).
#     * Generating specific prediction samples (e.g., based on ICU stays).
#     * **EHR Feature Engineering:** Transforming raw EHR events (diagnoses, procedures, etc.)
#         into the sequence format expected. Crucially, the original paper used a
#         dense `t x 76` matrix derived from specific variable selection,
#         discretization, normalization, and one-hot encoding. This model, adapted
#         for PyHealth, assumes the Task provides sequences of *integer indices*
#         (via `ehr_feature_key`) which are then fed into an `nn.Embedding` layer within
#         this model. A Task replicating the paper's exact `t x 76` input would
#         require different model input handling (no embedding layer).
#     * **CXR Image Handling:** Finding the relevant CXR image path corresponding to
#         each EHR sample, loading the image, applying necessary preprocessing
#         (resizing, normalization), and providing the image tensor (via `cxr_feature_key`).
#     * **Pairing Information:** Determining if a corresponding CXR image was found
#         for each EHR sample and providing a boolean flag (via `cxr_pair_key`). This
#         is essential for the LSTM fusion logic.
#     * **Label Generation:** Calculating the appropriate ground truth label based on
#         the specific prediction target (e.g., mortality, phenotyping).
#     * **Schema Definition:** Defining the `input_schema` and `output_schema` correctly
#         so the `SampleDataset` uses appropriate processors (e.g., `VocabProcessor`
#         for EHR indices, an `ImageProcessor` for CXR, `DefaultProcessor` or
#         `PytorchProcessor` for the pairing flag and labels).
#
# 2.  **`SampleDataset` Configuration:** The `SampleDataset` instance passed to this
#     model's `__init__` must be generated using the custom `MedFuseTask` described
#     above. This ensures the dataset contains the correct processed data fields
#     (EHR indices, CXR tensors, lengths, pairing flag, labels) under the
#     specified feature keys, and that the dataset's internal processors (e.g.,
#     vocabulary) are correctly fitted.
#
# **Assumptions made by this Model Code:**
# * `ehr_feature_key`: Refers to a field containing sequences of integer indices
#     representing EHR codes/events.
# * `cxr_feature_key`: Refers to a field containing preprocessed CXR image tensors
#     (batch x C x H x W).
# * `ehr_length_key`: Refers to a field containing the true lengths of the EHR
#     sequences before padding.
# * `cxr_pair_key`: Refers to a field containing a boolean tensor indicating
#     whether a paired CXR image is present for the sample.
# * The `SampleDataset` passed to `__init__` has been correctly built using a
#     compatible Task, allowing `get_input_voc_size` and `get_output_size` to function.
#
# This contribution focuses on providing the model architecture implementation
# within the PyHealth framework. Further work on the Task and Dataset components
# is required for end-to-end execution.
# --- End Comment Block ---


# --- Helper: Adapted LSTM Core (without final dense layer/activation) ---
class _EHR_LSTM_Core(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, layers=1, dropout=0.0, batch_first=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=layers,
            batch_first=batch_first, dropout=dropout if layers > 1 else 0.0 # Dropout only between LSTM layers
        )
        self.do = None
        if dropout > 0.0:
            self.do = nn.Dropout(dropout)
        self.feats_dim = hidden_dim

        self.initialize_weights()

    def initialize_weights(self):
        for model in self.modules():
            if type(model) in [nn.Linear]:
                nn.init.xavier_uniform_(model.weight)
                nn.init.zeros_(model.bias)
            elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
                nn.init.orthogonal_(model.weight_hh_l0)
                nn.init.xavier_uniform_(model.weight_ih_l0)
                nn.init.zeros_(model.bias_hh_l0)
                nn.init.zeros_(model.bias_ih_l0)

    def forward(self, x_packed): # Assumes input is already packed

        lstm_out_packed, (ht, _) = self.lstm(x_packed)
        # Get the hidden state of the last layer
        feats = ht[-1] # ht shape is (num_layers, batch, hidden_dim), get last layer
        if self.do is not None:
            feats = self.do(feats)
        # Return only the features (last hidden state)
        return feats

# --- Helper: Adapted CXR Core (without final classifier/activation) ---
class _CXR_Core(nn.Module):
    def __init__(self, vision_backbone_name='resnet34', pretrained=True):
        super().__init__()
        try:
            self.vision_backbone = getattr(torchvision.models, vision_backbone_name)(pretrained=pretrained)
        except AttributeError:
            raise ValueError(f"Unknown torchvision backbone: {vision_backbone_name}")

        classifiers = ['classifier', 'fc']
        self.feats_dim = -1
        for classifier in classifiers:
            cls_layer = getattr(self.vision_backbone, classifier, None)
            if cls_layer is None:
                continue
            if isinstance(cls_layer, nn.Module): # Check if it's a module
                # Get input features to the original classifier
                if hasattr(cls_layer, 'in_features'):
                     self.feats_dim = cls_layer.in_features
                elif isinstance(cls_layer, nn.Sequential) and hasattr(cls_layer[0], 'in_features'):
                     self.feats_dim = cls_layer[0].in_features
                else:
                    # Fallback: try getting output shape of layer before classifier
                    try:
                        # Get the layer name before the classifier
                        layer_name = list(self.vision_backbone._modules.keys())[-2]
                        output_shape = self.vision_backbone[layer_name].weight.shape[0]
                        self.feats_dim = output_shape # Approximate
                        print(f"Warning: Could not directly determine CXR feature dim, approximating as {self.feats_dim}")
                    except Exception:
                         raise ValueError(f"Could not determine feature dimension for backbone {vision_backbone_name}")


                # Replace classifier with Identity
                setattr(self.vision_backbone, classifier, nn.Identity())
                break

    def forward(self, x):
        visual_feats = self.vision_backbone(x)

        if visual_feats.dim() > 2:
            visual_feats = torch.flatten(visual_feats, 1)

        return visual_feats

# --- Main MedFuse Model (LSTM Fusion Only) ---
class MedFuse(BaseModel):
    """
    MedFuse model integrating EHR and CXR data for PyHealth using LSTM Fusion.

    This model architecture is adapted from the MedFuse paper, specifically
    implementing the LSTM fusion mechanism. It expects input data processed
    through a PyHealth `SampleDataset`, which should provide EHR data as
    sequences of indices, CXR data as preprocessed image tensors, EHR sequence
    lengths, and a boolean flag indicating CXR availability.

    Refer to the note at the top of this file for details on the required
    upstream Task and Dataset components needed for end-to-end execution.

    Args:
        dataset: A `pyhealth.datasets.SampleDataset` instance which has been
            processed by a compatible Task. The model uses this dataset to
            determine vocabulary sizes and output dimensions.
        ehr_feature_key: The key in the dataset samples corresponding to the
            EHR event sequences (expected as integer indices).
        cxr_feature_key: The key in the dataset samples corresponding to the
            preprocessed CXR image tensors.
        ehr_length_key: The key for the tensor containing true EHR sequence lengths.
            Defaults to "ehr_lengths".
        cxr_pair_key: The key for the boolean tensor indicating if a paired CXR
            image is available for the sample. Defaults to "has_cxr".
        ehr_embedding_dim: Dimension for the EHR code embeddings. Defaults to 128.
        ehr_lstm_hidden_dim: Hidden dimension size for the EHR LSTM branch.
            Defaults to 256.
        ehr_lstm_layers: Number of layers for the EHR LSTM branch. Defaults to 2.
        ehr_dropout: Dropout rate applied after the EHR LSTM. Defaults to 0.3.
        vision_backbone_name: Name of the torchvision model to use as the CXR
            backbone (e.g., 'resnet18', 'resnet34', 'densenet121'). Defaults to 'resnet34'.
        vision_pretrained: Whether to use pretrained weights (ImageNet) for the
            vision backbone. Defaults to True.
        fusion_lstm_hidden_dim: Hidden dimension size for the final fusion LSTM.
            Defaults to 256.
    """
    def __init__(
        self,
        dataset: SampleDataset,
        # Feature keys from the dataset
        ehr_feature_key: str,          # Key for EHR code sequences (e.g., "ehr_codes")
        cxr_feature_key: str,          # Key for CXR image tensor (e.g., "cxr_image")
        ehr_length_key: str = "ehr_lengths", # Key for EHR sequence lengths tensor
        cxr_pair_key: str = "has_cxr",     # Key for boolean tensor indicating CXR presence
        # EHR Branch config (map from args)
        ehr_embedding_dim: int = 128,
        ehr_lstm_hidden_dim: int = 256,   
        ehr_lstm_layers: int = 2,         
        ehr_dropout: float = 0.3,         
        # CXR Branch config (map from args)
        vision_backbone_name: str = 'resnet34', 
        vision_pretrained: bool = True,         
        # Fusion config (specific to LSTM fusion)
        fusion_lstm_hidden_dim: int = 256,  
    ):
        super().__init__(dataset) # Call parent class init first

        # Store feature keys
        if ehr_feature_key not in self.feature_keys:
            raise ValueError(f"EHR feature key '{ehr_feature_key}' not found in dataset features: {self.feature_keys}")
        if cxr_feature_key not in self.feature_keys:
            raise ValueError(f"CXR feature key '{cxr_feature_key}' not found in dataset features: {self.feature_keys}")

        self.ehr_feature_key = ehr_feature_key
        self.cxr_feature_key = cxr_feature_key
        self.ehr_length_key = ehr_length_key
        self.cxr_pair_key = cxr_pair_key

        # --- EHR Branch Initialization ---
        ehr_vocab_size = self.dataset.get_input_voc_size(self.ehr_feature_key)
        self.ehr_embedding = nn.Embedding(ehr_vocab_size, ehr_embedding_dim, padding_idx=0) # Assuming 0 is padding index
        self.ehr_branch = _EHR_LSTM_Core(
            input_dim=ehr_embedding_dim,
            hidden_dim=ehr_lstm_hidden_dim,
            layers=ehr_lstm_layers,
            dropout=ehr_dropout
        )
        ehr_feats_dim = self.ehr_branch.feats_dim # This is ehr_lstm_hidden_dim

        # --- CXR Branch Initialization ---
        self.cxr_branch = _CXR_Core(
            vision_backbone_name=vision_backbone_name,
            pretrained=vision_pretrained
        )
        cxr_feats_dim_raw = self.cxr_branch.feats_dim

        # --- Fusion Layer Initialization (LSTM Fusion Specific) ---
        # Project CXR features to match the EHR feature dimension
        self.cxr_projection = nn.Linear(cxr_feats_dim_raw, ehr_feats_dim)
        cxr_projected_dim = ehr_feats_dim # Dimension after projection

        # Fusion LSTM takes sequence of [EHR_feats, CXR_projected_feats]
        # Input dimension is the sum of the two feature dimensions
        fusion_lstm_input_dim = ehr_feats_dim + cxr_projected_dim
        self.fusion_lstm = nn.LSTM(
            fusion_lstm_input_dim,
            fusion_lstm_hidden_dim,
            batch_first=True,
            num_layers=1, # Keeping fusion LSTM simple for now
            dropout=0.0   # No dropout within single-layer LSTM
        )
        final_feature_dim = fusion_lstm_hidden_dim # Output of fusion LSTM

        # --- Final Classifier ---
        output_size = self.get_output_size() # Get output size from BaseModel based on dataset
        self.final_classifier = nn.Linear(final_feature_dim, output_size)


    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the MedFuse model (LSTM Fusion).

        Args:
            data: A dictionary containing the batch data, expected to have keys
                  matching those provided during initialization (e.g.,
                  ehr_feature_key, cxr_feature_key, ehr_length_key, cxr_pair_key).
                  EHR data should be integer indices, CXR data should be image tensors.

        Returns:
            A tensor containing the raw output logits from the final classifier.
        """
        # Extract data - Ensure keys match what your dataset provides!
        ehr_indices = data[self.ehr_feature_key]    # (batch, seq_len)
        cxr_image = data[self.cxr_feature_key]      # (batch, C, H, W)
        ehr_lengths = data[self.ehr_length_key]     # (batch,)
        # Get pairing information (essential for handling missing CXR in LSTM fusion)
        if self.cxr_pair_key not in data:
             raise KeyError(f"Required key '{self.cxr_pair_key}' not found in input data dictionary. "
                            "The dataset must provide CXR pairing information via this key.")
        has_cxr = data[self.cxr_pair_key].bool() # Ensure it's boolean (batch,)


        # --- EHR Branch ---
        ehr_embedded = self.ehr_embedding(ehr_indices) # (batch, seq_len, embedding_dim)
        # Pack sequence
        # Ensure lengths are on CPU for pack_padded_sequence
        # Clamp lengths to be at least 1 to avoid issues with empty sequences if they occur
        ehr_lengths_cpu = ehr_lengths.cpu().clamp(min=1)
        packed_ehr_embedded = pack_padded_sequence(
            ehr_embedded, ehr_lengths_cpu, batch_first=True, enforce_sorted=False
        )
        ehr_feats = self.ehr_branch(packed_ehr_embedded) # (batch, ehr_lstm_hidden_dim)

        # --- CXR Branch ---
        cxr_feats_raw = self.cxr_branch(cxr_image) # (batch, cxr_raw_feat_dim)
        # Project CXR features
        cxr_feats_projected = self.cxr_projection(cxr_feats_raw) # (batch, ehr_lstm_hidden_dim)

        # --- Handle Missing CXR Data ---
        # Zero out projected CXR features for samples that don't have a pair
        cxr_feats_final = cxr_feats_projected.clone() # Clone to avoid in-place modification issues
        cxr_feats_final[~has_cxr] = 0.0

        # --- LSTM Fusion Sequence Construction ---
        # Create the 2-step sequence input: [EHR_only_step, Combined_step]
        # Step 1 features: EHR features + Zeros where CXR features would be
        step1_features = torch.cat([ehr_feats, torch.zeros_like(cxr_feats_final)], dim=1)
        # Step 2 features: EHR features + (Potentially Zeroed) Projected CXR features
        step2_features = torch.cat([ehr_feats, cxr_feats_final], dim=1)

        # Stack to form sequence: (batch, seq_len=2, feature_dim)
        # feature_dim = ehr_lstm_hidden_dim + ehr_lstm_hidden_dim
        lstm_input_seq = torch.stack([step1_features, step2_features], dim=1)

        # Determine sequence lengths for fusion LSTM based on CXR presence
        # Length is 1 if no CXR (only step1 matters), length is 2 if CXR is present
        # Use .long() for lengths expected by pack_padded_sequence
        fusion_lengths = torch.ones(self.batch_size, device=self.device, dtype=torch.long) + has_cxr.long()
        # Ensure lengths are on CPU and > 0
        fusion_lengths_cpu = fusion_lengths.cpu().clamp(min=1)

        # Pack the fusion sequence
        packed_fusion_input = pack_padded_sequence(
            lstm_input_seq, fusion_lengths_cpu, batch_first=True, enforce_sorted=False
        )

        # --- Pass through Fusion LSTM ---
        _, (ht, _) = self.fusion_lstm(packed_fusion_input)
        # ht shape is (num_fusion_lstm_layers, batch, fusion_lstm_hidden_dim)
        final_fused_features = ht[-1] # Get hidden state from last layer

        # --- Classification ---
        logits = self.final_classifier(final_fused_features)

        return logits

