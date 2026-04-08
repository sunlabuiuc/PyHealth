"""Implementation of Alzheimer's Disease Classification Model for ADNI Data.

Author: Bryan Lau (bryan16@illinois.edu)
Description:
    Implementation of the Alzheimer's Disease classification model described in the 
    paper "On the Design of Convolutional Neural Networks for Automatic Detection 
    of Alzheimer's Disease" by Liu et al. (https://arxiv.org/abs/1911.03740)
"""
import torch
import torch.nn as nn

from typing import Dict
from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel


class AlzheimersDiseaseCNN(BaseModel):
    """AlzheimersDiseaseCNN model for detection of Alzheimer's Disease using 
    ADNI datasets.

    Author: Bryan Lau

    Implements the architecture described in "On the Design of Convolutional 
    Neural Networks for Automatic Detection of Alzheimer's Disease" by Liu et 
    al. (https://arxiv.org/abs/1911.03740).

    The model uses MRI brain scans to classify subjects as:

    - Cognitively Normal (CN)
    - Mild Cognitive Impairment (MCI)
    - Alzheimer's Disease (AD)

    The architecture consists of four convolutional blocks as defined in the 
    paper, and can optionally incorporate the patient's age via positional 
    encoding.

    Args:
        dataset (SampleDataset): Pyhealth dataset containing:
            - input_schema: "image"
            - output_schema: "label"
        width_factor (int, optional): Channel multiplier for convolutional layers. 
            Defaults to 4.
        dropout (float, optional): Dropout value. Defaults to None.
        use_age (bool, optional): Whether to incorporate patient's age via positional 
            encoding. Liu et al. show that including age provides a minor 
            performance improvement. Defaults to False.
        norm_method (str, optional): Use either "instance" (default) or "batch" 
            normalization.
        class_weights (tensor): Weights for the loss function to handle class 
            imbalance. Defaults to None.

    Example:
        >>> from pyhealth.datasets import ADNIDataset
        >>> from pyhealth.tasks import AlzheimersDiseaseClassification
        >>> adni_dataset = ADNIDataset(root="/path/to/adni_data", dev=True)
        >>> ad_task = AlzheimersDiseaseClassification()
        >>> samples = adni_dataset.set_task(ad_task)
        >>> model = AlzheimersDiseaseCNN(
        ...     dataset=adni_dataset,
        ...     width_factor=4,
        ...     dropout=0.5,
        ...     use_age=True,
        ... )
    """

    def __init__(
        self,
        dataset: SampleDataset,
        width_factor=4,
        dropout=0.2,
        use_age=False,
        use_gender=False,
        norm_method="instance",
        class_weights=None,
        **kwargs
    ):
        """Initialize the model

        Args:
            dataset (SampleDataset): Pyhealth dataset containing MRI image and 
                labels.
            width_factor (int): Channel multiplier for convolutional layers. 
                Defaults to 4.
            dropout (float): Dropout value. Defaults to None.
            use_age (bool): Whether to encode ages. Defaults to False.
            use_gender (bool): Whether to encode genders. Defaults to False.
            norm_method (str): Use either "instance" (default) or "batch" 
                normalization.
            class_weights (tensor): Class weights for the loss function. Defaults 
                to None.
        """

        # Call superclass initialization first
        super(AlzheimersDiseaseCNN, self).__init__(dataset=dataset, **kwargs)

        # Store some values
        self.label_key = self.label_keys[0]
        self.input_channels = 1
        self.dropout = dropout
        self.use_age = use_age
        self.use_gender = use_gender
        self.norm_method = norm_method
        self.num_classes = self.get_output_size()

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        ####### CNN Layers #######

        # Input
        # Output size: (96, 96, 96)
        #   k: kernel size
        #   c: number of channels as a multiple of the widening factor
        #   f: widening factor
        #   p: padding size
        #   s: stride
        #   d: dilation

        # Block 1
        # k1-c4-f-p0-s1-d1
        # Conv3D    -> (96, 96, 96)
        # MaxPool3D -> 47x47x47 (approx half)
        self.block1 = nn.Sequential(
            nn.Conv3d(self.input_channels, 4 * width_factor,
                      kernel_size=1, stride=1, padding=0, dilation=1),
            self._get_normalization_layer(4 * width_factor),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2)
        )

        # Block 2
        # k3-c32-f-p0-s1-d2
        # Conv3D    -> 43x43x43
        # MaxPool3D -> 21x21x21 (approx half)
        self.block2 = nn.Sequential(
            nn.Conv3d(4 * width_factor, 32 * width_factor,
                      kernel_size=3, stride=1, padding=0, dilation=2),
            self._get_normalization_layer(32 * width_factor),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2)
        )

        # Block 3
        # k5-c64-f-p2-s1-d2
        # Conv3D    ->  17x17x17
        # MaxPool3D ->  8x8x8 (approx half)
        self.block3 = nn.Sequential(
            nn.Conv3d(32 * width_factor, 64 * width_factor,
                      kernel_size=5, stride=1, padding=2, dilation=2),
            self._get_normalization_layer(64 * width_factor),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2)
        )

        # Block 4
        # k3-c64-f-p1-s1-d2
        # Conv3D    -> 6x6x6
        # MaxPool3D -> 5x5x5
        self.block4 = nn.Sequential(
            nn.Conv3d(64 * width_factor, 64 * width_factor,
                      kernel_size=3, stride=1, padding=1, dilation=2),
            self._get_normalization_layer(64 * width_factor),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=5, stride=2)
        )

        # Table 2 doesn't explain how the model reduces 5x5x5 for the
        # first fully connected layer, so here we opt to reduce the
        # dimensionality using global average pooling
        self.global_pooling = nn.AdaptiveAvgPool3d(1)

        # Fully connected layers
        # FC1       1024
        # FC2       3  (final classification output)
        self.fc1 = nn.Linear(64 * width_factor, 1024)
        self.fc2 = nn.Linear(1024, self.num_classes)

        # Dropout
        if self.dropout:
            self.do = nn.Dropout(p=self.dropout)

        ####### Age Encoding #######
        # Initialize age encoding per Appendix A
        if self.use_age:

            # Dimensionality of age encoding
            pos_enc_dim = 512

            # Fix age values from 0 to 120 years old rounding
            # to 0.5 decimals (240 possible age values)
            ages = torch.arange(0, 120.5, 0.5)
            num_ages = len(ages)

            # Initialize positional encoding table
            age_encoding = torch.zeros(num_ages, pos_enc_dim)

            # Compute age positional encodings
            position = ages.unsqueeze(1)  # -> (240, 1)
            div_term = torch.exp(torch.arange(0, pos_enc_dim, 2).float(
            ) * -(torch.log(torch.tensor(10000.0)) / pos_enc_dim))
            age_encoding[:, 0::2] = torch.sin(position * div_term)
            age_encoding[:, 1::2] = torch.cos(position * div_term)

            # Register PyTorch buffer which is saved with
            # the model but isn't a trainable parameter
            self.register_buffer("age_encoding_table", age_encoding)

            # Age transformation layers
            # Output size:
            #   Linear      (512 -> 512)
            #   LayerNorm   (512)
            #   Linear      (512 -> 1024)
            self.age_fc1 = nn.Linear(pos_enc_dim, 512)
            self.age_norm = nn.LayerNorm(512)
            self.age_fc2 = nn.Linear(512, 1024)

        # Gender embedding layer
        if self.use_gender:
            self.gender_embed = nn.Embedding(2, 1024)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        The expected keys in the arguments are:

        image (torch.Tensor): MRI image with shape (batch_size, 1, D, H W)
        age (torch.Tensor): Optional patient ages with shape (batch_size, 1)
        label (torch.Tensor): Diagnostic labels associated with each patient, 
            converted to numeric values, i.e. CN, MCI, AD -> 0, 1, 2

        Args:
            **kwargs: Arguments from Pyhealth dataloader

        Returns:
            Dictionary (Dict[str, torch.Tensor]) containing:
                "loss"
                "y_prob"
                "y_true"
                "logit"
        """

        # Extract image from kwargs
        images = kwargs["image"]
        if isinstance(images, list):
            x = torch.stack(images, dim=0).to(self.device)
        else:
            x = images.to(self.device)

        # Extract additional attributes
        age = kwargs.get("age", None)
        gender = kwargs.get("gender", None)

        # Pass MRI image data through CNN blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Apply global average pooling to reduce dimensions for fc1 (to 1x1x1)
        x = self.global_pooling(x)

        # Flatten for fully connected layer -> (batch_size, 64*width_factor)
        x = torch.flatten(x, 1)

        # Fully connected layer 1
        x = self.fc1(x)
        x = nn.functional.relu(x)

        # Add age data (optionally)
        if self.use_age:

            # Round ages to nearest 0.5 and clamp to range [0, 120]
            age_rounded = torch.round(age * 2) / 2
            age_rounded = torch.clamp(age_rounded, 0, 120)

            # Convert age to indices for age encoding table
            age_indices = (age_rounded * 2.0).long()

            # Perform lookup of positional encodings
            age_pos_enc = self.age_encoding_table[age_indices]

            # Pass age data through age transformation layers
            age_embeds = self.age_fc1(age_pos_enc)
            age_embeds = self.age_norm(age_embeds)
            age_embeds = self.age_fc2(age_embeds)

            # Add age embeddings to CNN features (per Appendix A)
            x = x + age_embeds

        # Embed gender (optionally)
        if self.use_gender:

            gender_tensor = gender.to(self.device).long()
            x = x + self.gender_embed(gender_tensor)

        # Apply dropout
        if self.dropout:
            x = self.do(x)

        # Compute final classification logits
        logits = self.fc2(x)

        # Compute loss and probabilities
        y_true = kwargs[self.label_key].to(self.device)
        loss = torch.nn.functional.cross_entropy(
            logits, y_true, weight=self.class_weights)
        y_prob = self.prepare_y_prob(logits)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }

    def _get_normalization_layer(self, num_channels):
        """Get an InstanceNorm3d or BatchNorm3d layer according to the 
        norm_method parameter.

        Args:
            num_channels: Number of channels to initialize.

        Returns:
            InstanceNorm3d or BatchNorm3d instance.
        """
        if self.norm_method == "instance":
            return nn.InstanceNorm3d(num_channels)
        elif self.norm_method == "batch":
            return nn.BatchNorm3d(num_channels)
        else:
            raise ValueError(f"Unknown normalization type: {self.norm_method}")
