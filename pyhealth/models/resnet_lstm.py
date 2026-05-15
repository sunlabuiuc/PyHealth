"""
PyHealth task for extracting features with STFT and Frequency Bands using the Temple University Hospital (TUH) EEG Seizure Corpus (TUSZ) dataset V2.0.5.

Dataset link:
    https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml

Dataset paper:
    Vinit Shah, Eva von Weltin, Silvia Lopez, et al., “The Temple University Hospital Seizure Detection Corpus,” arXiv preprint arXiv:1801.08085, 2018. Available: https://arxiv.org/abs/1801.08085

Dataset paper link:
    https://arxiv.org/abs/1801.08085

Author:
    Fernando Kenji Sakabe (fks@illinois.edu), 
    Jesica Hirsch (jesicah2@illinois.edu), 
    Jung-Jung Hsieh (jhsieh8@illinois.edu)
"""
import torch
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from .eeg_feature_extractors import FeatureExtractorManager, RESNET_LSTM


class BasicBlock(nn.Module):
    """Residual building block for the 2D convolutional ResNet encoder.

    This block applies two (1×9) convolutions with batch normalization and a
    residual (shortcut) connection. The shortcut uses a (1×1) convolution to
    match channel dimensions when they differ.

    Args:
        in_planes: number of input channels.
        planes: number of output channels.
        stride: stride applied to the first convolution and the shortcut
            projection. Default is 1.
    """

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=(1,9), stride=(1,stride), padding=(0,4), bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=(1,9), stride=1, padding=(0,4), bias=False),
            nn.BatchNorm2d(planes),
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=(1,stride), bias=False),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.net(x)
        out += self.shortcut(x)
        return torch.relu(out)


class ResNetLSTM(BaseModel):
    """2-D ResNet + LSTM model for multivariate time-series classification.

    The model processes multi-channel signals (e.g., EEG/ECG) through an optional
    feature extractor, a residual convolutional encoder (ResNet-style), and a
    multi-layer LSTM. The final LSTM hidden state is passed to a classification
    head that produces output logits.

    Pipeline:
        1. (Optional) A ``feature_extractor`` (e.g. spectrogram or wavelet
        transform) converts the raw signal into a 2-D representation.
        If ``encoder=None``, the raw signal is used directly.
        2. ``feature_transformer`` reshapes/transforms the features into a format
        suitable for convolutional processing.
        3. ``feature_extractor_cnn`` applies initial Conv2d layers to project the
        signal into a feature map (exact architecture depends on the
        FeatureExtractorManager implementation).
        4. Three residual stages (``BasicBlock``) progressively refine the
        representation and increase channel depth (typically 64 → 128 → 256),
        optionally reducing temporal resolution via stride.
        5. Adaptive average pooling collapses the spatial dimension to 1×1,
        yielding a per-frame feature vector.
        6. The LSTM models temporal dependencies across frames; only the last
        output step is retained.
        7. A two-layer fully-connected head maps the LSTM output to
        ``output_dim`` logits.

    Args:
        dataset (SampleDataset):
            Dataset with fitted input and output processors.

        encoder (str or None):
            Name of the feature extractor backbone.
            If None, no pre-trained feature extraction is applied.

        num_layers (int):
            Number of LSTM layers.

        output_dim (int):
            Number of output classes or regression targets.

        batch_size (int):
            Batch size used to initialise the LSTM hidden state.
            NOTE: This must match the actual batch size during forward pass.

        dropout (float):
            Dropout probability applied after activations and between LSTM layers.
            Default is 0.5.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {
        ...         "patient_id": "patienta",
        ...         "record_id": "s001_2003",
        ...         "signal_file": "dev/patienta/s001_2003/01_tcp_ar/patienta_s001_2003.edf",
        ...     },
        ...     {
        ...         "patient_id": "patientb",
        ...         "record_id": "s007_2014",
        ...         "signal_file": "eval/patientb/s007_2014/01_tcp_ar/patientb_s007_2014.edf",
        ...     },
        ... ]
        >>>
        >>> # dataset
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={
        ...         "signal": "tensor",
        ...     },
        ...     output_schema={
        ...         "label": "tensor"
        ...         "label_bitgt_1": "tensor",
        ...         "label_bitgt_2": "tensor",
        ...         "label_name": "text",
        ...     },
        ... )
        >>>
        >>> # data loader
        >>> from pyhealth.datasets import get_dataloader
        >>> train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>>
        >>> model = ResNetLSTM(
        ...     dataset=dataset,
        ...     encoder=None,
        ...     num_layers=2,
        ...     output_dim=2,
        ...     batch_size=32,
        ... )

    Forward:
        Input:
            x: tensor of shape [batch_size, sequence_length, channels]

        Output:
            output:
                Tensor of shape [batch_size, output_dim] (logits)

            hidden:
                Tuple (h_n, c_n) where each has shape
                [num_layers, batch_size, hidden_dim]

    Notes:
        - Input is internally permuted to [batch, channels, time].
        - CNN architecture depends on FeatureExtractorManager.
        - The model assumes fixed batch size due to hidden state initialization.
    """
    def __init__(
        self,
        dataset: SampleDataset,
        encoder: str,
        num_layers: int,
        output_dim: int,
        batch_size: int,
        dropout: float = 0.5,
    ):
        super().__init__(dataset=dataset)

        self.encoder = encoder
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.hidden_dim = 256

        self.feature_manager = FeatureExtractorManager(model=RESNET_LSTM, encoder=self.encoder)
        self.feature_extractor = self.feature_manager.get_feature_extractor()
        self.feature_transformer = self.feature_manager.transform_features
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.feature_extractor_cnn = self.feature_manager.get_feature_extractor_cnn(
            activation        = self.activation,
            dropout           = self.dropout,
        )

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  num_blocks=2, stride=1)  
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)  
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)  

        self.agvpool = nn.AdaptiveAvgPool2d((1,1))

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.BatchNorm1d(64),
            self.activation,
            nn.Linear(64, self.output_dim),
        )

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Build a residual stage consisting of ``num_blocks`` BasicBlocks.

        The first block uses the supplied ``stride`` to optionally downsample;
        all subsequent blocks use stride=1. ``self.in_planes`` is updated in
        place so the next stage picks up the correct channel count.

        Args:
            planes: number of output channels for every block in this stage.
            num_blocks: number of BasicBlocks to stack.
            stride: stride applied only to the first block of the stage.

        Returns:
            nn.Sequential containing the stacked BasicBlocks.
        """
        layers = []
        strides = [stride] + [1]*(num_blocks-1)

        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Forward propagation.

        Args:
            x: input tensor of shape [batch size, sequence length, channels].

        Returns:
            output: logit tensor of shape [batch size, output_dim].
            hidden: tuple of final LSTM hidden and cell states, each of shape
                [num_layers, batch size, hidden_dim].
        """
        x = x.permute(0, 2, 1)

        if self.feature_extractor is not None:
            x = self.feature_extractor(x)
        x = self.feature_transformer(x)
        x = self.feature_extractor_cnn(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.agvpool(x)
        x = torch.squeeze(x, 2)
        x = x.permute(0, 2, 1)

        batch_size = x.size(0)
        self.hidden = (
            x.new_zeros(self.num_layers, batch_size, self.hidden_dim),
            x.new_zeros(self.num_layers, batch_size, self.hidden_dim)
        )

        output, hidden = self.lstm(x, self.hidden)
        output = output[:, -1, :]

        output = self.classifier(output)

        return output, hidden