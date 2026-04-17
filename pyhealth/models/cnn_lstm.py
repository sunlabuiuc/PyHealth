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
from .eeg_feature_extractors import FeatureExtractorManager, CNN_LSTM


class CNNLSTM(BaseModel):

    """CNN + LSTM model for EEG classification in PyHealth 2.0.

    The model processes raw multi-channel EEG signals through an optional
    feature extractor, a convolutional encoder, and a multi-layer LSTM.
    The final LSTM output is passed to a small fully connected classification head.

    Pipeline:
        1. (Optional) A ``feature_extractor`` (e.g. spectrogram or wavelet
        transform) converts the raw EEG signal into a 2-D representation.
        If ``encoder=None``, the raw signal is used directly.
        2. ``feature_transformer`` reshapes/transforms the features into a format
        suitable for convolutional processing.
        3. ``feature_extractor_cnn`` applies a Conv2d-based encoder to extract
        higher-level representations (exact architecture depends on the
        FeatureExtractorManager implementation).
        4. Adaptive average pooling collapses the spatial dimension to 1×1,
        yielding a per-frame feature vector.
        5. The LSTM models temporal dependencies across frames; only the last
        output step is retained.
        6. A two-layer fully-connected head maps the LSTM output to
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
            NOTE: This must match the actual batch size used during forward pass.

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
        >>> model = CNNLSTM(
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
        - The CNN architecture depends on FeatureExtractorManager.
        - The model assumes fixed batch size due to hidden state initialization.
    """
    def __init__(
            self,
            dataset: SampleDataset,
            encoder: str,
            num_layers: int,
            output_dim: int,
            batch_size: str,
            dropout: float = 0.5,
        ):
        super(CNNLSTM, self).__init__(
            dataset=dataset,
        )
        self.encoder    = encoder
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.batch_size = batch_size

        self.feature_manager = FeatureExtractorManager(model=CNN_LSTM, encoder=self.encoder)
        self.feature_extractor = self.feature_manager.get_feature_extractor()
        self.feature_transformer = self.feature_manager.transform_features
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.feature_extractor_cnn = self.feature_manager.get_feature_extractor_cnn(
            activation        = self.activation,
            dropout           = self.dropout,
        )

        self.agvpool = nn.AdaptiveAvgPool2d((1,1))

        self.hidden_dim = 256
        self.lstm = nn.LSTM(
                input_size=self.feature_manager.output_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=dropout)

        self.classifier = nn.Sequential(
                nn.Linear(in_features=self.hidden_dim, out_features=64, bias=True),
                nn.BatchNorm1d(64),
                self.activation,
                nn.Linear(in_features=64, out_features=self.output_dim, bias=True),
        )

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

        x = self.agvpool(x)
        x = torch.squeeze(x, 2)
        x = x.permute(0, 2, 1)

        self.hidden = (
            (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
             torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
        )

        output, self.hidden = self.lstm(x, self.hidden)
        output = output[:, -1, :]

        output = self.classifier(output)

        return output, self.hidden