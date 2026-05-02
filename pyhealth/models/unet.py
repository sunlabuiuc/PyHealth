"""
U-Net architecture for image segmentation.

Paper: Ronneberger et al. U-Net: Convolutional Networks for Biomedical
Image Segmentation. MICCAI 2015.
Paper Link: https://arxiv.org/abs/1505.04597

Improved architecture following principles from:
Paper: Isensee et al. nnU-Net: a self-configuring method for deep learning-based
biomedical image segmentation. Nature Methods 2021.
Paper Link: https://arxiv.org/abs/1809.10486
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class DoubleConv(nn.Module):
    """
    (convolution => instance norm => LeakyReLU) * 2 with Residual Connection.
    Input and output feature volume without striding is kept same dimension
    by using padding; kernel sizes are always 3.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mid_channels (int | None, optional): Number of middle channels.
            Defaults to None (uses out_channels).
        stride (int, optional): Stride for the first convolution. Defaults to 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        stride: int = 1,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        residual = self.shortcut(x)
        out = self.double_conv(x)
        out += residual
        # Use leaky ReLU to prevent dead neurons.
        return F.leaky_relu(out, negative_slope=0.01, inplace=True)


class Up(nn.Module):
    """
    Upscaling then concatenation with skip connection followed by double conv.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        This method performs upscaling, calculates any spatial dimension
        mismatches between the upscaled features and the skip connection
        features (often due to non-power-of-two input sizes), pads the
        upscaled features to match the skip connection, and concatenates
        them before the final double convolution.

        Args:
            x1 (torch.Tensor): Input tensor to be upscaled.
            x2 (torch.Tensor): Skip connection tensor from the encoder.

        Returns:
            torch.Tensor: Upscaled output tensor.
        """
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(BaseModel):
    """
    U-Net model for image segmentation.

    For model inputs:
        - An image tensor of dimension (B, C, H, W), at key "image" by default
        - An optional mask tensor of shape (B, 1, H, W) or (B, H, W),
            at "mask" by default
    The model produces per-pixel classifications of shape (B, n_classes, H, W).

    This is a fully convolutional version of U-Net, as per the original paper,
    with some improvements. The first half of the network is a spatial downsampling
    route, where features are successively sampled and extracted to a spatially
    smaller output tensor, which has the effect of merging features across the whole
    image. By the time features propagate to the "bottom" of the U-Net, convolutions
    essentially attend to all spatial features across the entire input image.
    From there, an upsampling path projects global features back to the original
    input image space, and concatenates features from "skip" connections in order
    to retain locality in features. This allows each spatial location in the final
    feature volume to combine both local and global features.

    Args:
        dataset (SampleDataset): The dataset for training; this model was
            implemented with a CXRSegmentationDataset in mind.
        base_filters (int, optional): Number of filters in the first layer.
            Defaults to 64.
        depth (int, optional): Number of downsampling steps. Defaults to 4.
        n_channels (int | None, optional): Number of input channels.
            If None, it will be inferred from the dataset. Defaults to None.
        n_classes (int, optional): Number of output classes.
            If 1, binary segmentation with sigmoid is used.
            If > 1, multiclass segmentation with softmax is used.
            Defaults to 1.

    Usage example for training:
        >>> import torch
        >>> from pyhealth.datasets import CXRSegmentationDataset, get_dataloader
        >>> from pyhealth.tasks import CXRSegmentationTask
        >>> from pyhealth.models import UNet
        >>> # load the dataset and apply the task
        >>> dataset = CXRSegmentationDataset(root="/path/to/dataset", dev=True)
        >>> samples = dataset.set_task(CXRSegmentationTask())
        >>> # initialize the model
        >>> model = UNet(dataset=samples)
        >>> # single forward pass
        >>> loader = get_dataloader(samples, batch_size=1)
        >>> data_batch = next(iter(loader))
        >>> results = model(**data_batch)
        >>> results["y_prob"].shape
        torch.Size([1, 1, 224, 224])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        base_filters: int = 64,
        depth: int = 4,
        n_channels: int | None = None,
        n_classes: int = 1,
        **kwargs,
    ):
        super(UNet, self).__init__(dataset=dataset)
        self.n_classes = n_classes
        self.base_filters = base_filters
        self.depth = depth

        assert len(self.feature_keys) == 1, "Only one image feature is supported"
        assert len(self.label_keys) == 1, "Only one mask label is supported"
        self.image_key = self.feature_keys[0]
        self.mask_key = self.label_keys[0]

        if n_channels is None:
            self.n_channels = self._inferred_input_channels()
        else:
            self.n_channels = n_channels

        filters = [base_filters * (2**i) for i in range(depth + 1)]

        self.inc = DoubleConv(self.n_channels, filters[0])

        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(DoubleConv(filters[i], filters[i + 1], stride=2))

        self.ups = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.ups.append(Up(filters[i], filters[i - 1]))

        # Final 1x1 convolution layer just projects each spatial location
        # to one channel; i.e. the classification.
        self.outc = nn.Conv2d(filters[0], n_classes, kernel_size=1)

        # Set mode to enable automatic metrics selection in Trainer
        self.mode = "segmentation"

    def _inferred_input_channels(self) -> int:
        """
        Infer input channels from the dataset.

        Returns:
            int: Number of channels; if not inferrable, returns 1.
        """
        for sample in self.dataset:
            if self.image_key in sample:
                image = sample[self.image_key]
                if isinstance(image, torch.Tensor):
                    return image.shape[0]
                elif isinstance(image, (list, tuple)) and len(image) > 0:
                    # Case where it might be (processor_name, tensor)
                    if isinstance(image[1], torch.Tensor):
                        return image[1].shape[0]
        return 1

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """
        Forward pass, producing per-pixel predicted class logits
        and softmax probabilities. If a mask is provided, loss
        is also computed here using a linear combination of DICE
        and CE loss.

        Args:
            **kwargs: Input features and labels.
                - The key specified in `self.image_key` (typically "image") is
                  required and should be a tensor of shape (B, C, H, W).
                - The key specified in `self.mask_key` (typically "mask") is
                  optional and should be a tensor of shape (B, 1, H, W) or (B, H, W).
                  If provided, the loss will be computed.

        Returns:
            dict[str, torch.Tensor]: A dictionary with keys:
                - logit: predicted logits of shape (B, n_classes, H, W).
                - y_prob: predicted probabilities of shape (B, n_classes, H, W).
                - loss: scalar loss tensor (if mask was provided).
                - y_true: true mask tensor (if mask was provided).
        """
        x = kwargs[self.image_key]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device)
        else:
            x = x.to(self.device)

        # Encoder, spatial downsampling path
        x_list = [self.inc(x)]
        for down in self.downs:
            x_list.append(down(x_list[-1]))

        # Decoder, spatial upsampling with skip connections
        x = x_list[-1]
        for i, up in enumerate(self.ups):
            x = up(x, x_list[-(i + 2)])

        logits = self.outc(x)

        if self.n_classes == 1:
            y_prob = torch.sigmoid(logits)
        else:
            y_prob = F.softmax(logits, dim=1)

        results = {
            "logit": logits,
            "y_prob": y_prob,
        }

        # If masks were provided, we should also compute loss here
        if self.mask_key in kwargs:
            y_true = kwargs[self.mask_key]
            if not isinstance(y_true, torch.Tensor):
                y_true = torch.tensor(y_true, device=self.device)
            else:
                y_true = y_true.to(self.device)

            # DICE Loss calculation
            # Use y_prob for dice loss to allow gradients to flow
            if self.n_classes == 1:
                y_true_onehot = y_true.float()
            else:
                # multiclass segmentation: one-hot encode y_true
                # y_true shape: (B, 1, H, W) or (B, H, W)
                if y_true.dim() == 4:
                    y_true_indices = y_true.squeeze(1).long()
                else:
                    y_true_indices = y_true.long()
                y_true_onehot = F.one_hot(y_true_indices, num_classes=self.n_classes)
                # y_true_onehot shape: (B, H, W, n_classes) -> (B, n_classes, H, W)
                y_true_onehot = y_true_onehot.permute(0, 3, 1, 2).float()

            dims = (0, 2, 3)
            intersection = torch.sum(y_true_onehot * y_prob, dim=dims)
            cardinality = torch.sum(y_true_onehot + y_prob, dim=dims)
            dice_loss = (1.0 - (2.0 * intersection + 1e-7) / (cardinality + 1e-7)).mean()

            if self.n_classes == 1:
                # Binary class segmentation; use binary_cross_entropy_with_logits for
                # numerical stability
                ce_loss = F.binary_cross_entropy_with_logits(logits, y_true.float())
            else:
                # Multiclass segmentation; use the generalized cross entropy loss.
                if y_true.dim() == 4 and y_true.shape[1] == 1:  # (B, 1, H, W)
                    ce_loss = F.cross_entropy(logits, y_true.squeeze(1).long())
                else:  # (B, H, W)
                    ce_loss = F.cross_entropy(logits, y_true.long())

            # Many modern U-Nets use a combination of DICE and standard CE loss.
            results["loss"] = ce_loss + dice_loss
            results["y_true"] = y_true

        return results
