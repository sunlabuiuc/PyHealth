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
import torchaudio

import torch
from torch import nn

class PsdFeatureExtractor(nn.Module):
    """Frequency Bands feature extractor for EEG classification.

    The Frequency Bands with STFT extracts averaged features with 7 predetermined frequency bands.

    Pipeline:
        1. Transform input with STFT.
        2. Extract features using 7 frequency bands.
        3. Stack the results.

    Args:
        sample_rate (int):
            The required sample rate. Default to 200.

        frame_length (int):
            The window size for signals. Default to 16.

        frame_shift (int):
            The window shift for signals. Default to 8.

        feature_extract_by (str):
            Use 'kaldi' if the platform is linux or darwin. Default to 'kaldi'.

    Examples:
        >>> class TestModel(BaseModel):
        ...     def __init__():
        ...         self.feature_extractor = FEATURE_EXTRACTORS['psd']
        ...
        ...     def forward(self, x: torch.Tensor):
        ...         output = self.feature_extractor(x)
        ...         return output
        >>>
        >>> model = TestModel()
        >>>
        >>> logits, _ = model(x)
    """
    def __init__(self,
        sample_rate: int = 200,
        frame_length: int = 16,
        frame_shift: int = 8,
        feature_extract_by: str = 'kaldi',
    ) -> None:
        super(PsdFeatureExtractor, self).__init__()

        self.sample_rate = sample_rate
        self.feature_extract_by = feature_extract_by.lower()
        self.freq_resolution = 1
        self.n_fft = self.freq_resolution*self.sample_rate
        self.hop_length = frame_shift
        self.frame_length = frame_length

        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.frame_length
        ) if feature_extract_by == 'kaldi' else torch.stft(
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.frame_length,
            window=torch.hamming_window(self.frame_length),
            center=False,
            normalized=False,
            onesided=True
        )
        
    def psd(self, amp: torch.Tensor, begin: int, end: int) -> torch.Tensor:
        """Returns calculation for psd."""
        return torch.mean(amp[begin*self.freq_resolution:end*self.freq_resolution], 0)
        
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward propagation.

        Args:
            x: input tensor of shape [batch size, channels, sequence length].

        Returns:
            output: stacked input of shape [batch_size, channels, 7 (frequency bins), time_frames].
        """
        psds_batch = []

        for signals in batch:
            psd_sample = []
            for signal in signals:
                stft = self.stft(signal)
                amp = (torch.log(torch.abs(stft) + 1e-10))

                psd1 = self.psd(amp,0,4)
                psd2 = self.psd(amp,4,8)
                psd3 = self.psd(amp,8,12)
                psd4 = self.psd(amp,12,30)
                psd5 = self.psd(amp,30,50)
                psd6 = self.psd(amp,50,70)
                psd7 = self.psd(amp,70,100)
                
                psds = torch.stack((psd1, psd2, psd3, psd4, psd5, psd6, psd7))
                psd_sample.append(psds)

            psds_batch.append(torch.stack(psd_sample))

        return torch.stack(psds_batch)


CNN_LSTM = 'cnn_lstm'
RESNET_LSTM = 'resnet_lstm'
RAW = 'raw'
PSD = 'psd'

FEATURE_EXTRACTORS = nn.ModuleDict([
    [ RAW, None ],
    [ PSD, PsdFeatureExtractor() ],
])

CONV2D = {
    RAW: {
        'in_1': 1,   'out_1': 64,   'kernel_1': (1,51), 'stride_1': (1,4), 'padding_1': (0,25),
        'in_2': 64,  'out_2': 128,  'kernel_2': (1,21), 'stride_2': (1,2), 'padding_2': (0,10),
        'in_3': 128, 'out_3': 256,  'kernel_3': (1,9),  'stride_3': (1,2), 'padding_3': (0,4),
    },
    PSD: {
        'in_1': 1,   'out_1': 64,   'kernel_1': (7,21), 'stride_1': (7,2), 'padding_1': (0,10),
        'in_2': 64,  'out_2': 128,  'kernel_2': (1,21), 'stride_2': (1,2), 'padding_2': (0,10),
        'in_3': 128, 'out_3': 256,  'kernel_3': (1,9), 'stride_3': (1,1), 'padding_3': (0,4),
    }
}

MAX2D = {
    RAW: { 'kernel': (1,4), 'stride': (1,4) },
    PSD: { 'kernel': (1,2), 'stride': (1,2) },
}

CONV = 'conv'
MAXPOOL = 'maxpool'
ORDER2D = {
    CNN_LSTM: {
        RAW: [ CONV, MAXPOOL, CONV, CONV ],
        PSD: [ CONV, CONV, MAXPOOL, CONV ],
    },
    RESNET_LSTM: {
        RAW: [ CONV, MAXPOOL ],
        PSD: [ CONV, MAXPOOL ],
    },
}

class FeatureExtractorManager:
    """Manager for any feature-extractor-related tasks in EEG classification.

    This class manages the initializations and setups for each feature extractor.
    It allows for easy recombination between feature extractors and different models.

    Args:
        encoder,
        output_dim: int = 1,
        model (str):
            Model type is required to determine the cnn sequence for feature extractor.
            Supported types: CNN_LSTM, RESNET_LSTM

        encoder (str):
            Encoder type is required to determine the class of feature extractor, input transformation, 
            and the order and size of input, output, kernels, strides, and paddings in cnn.
            Supported types: RAW, PSD
        
        output_dim (int):
            The dimension of outputs for referencing in the subsequent steps.

    Examples:
        >>> self.feature_manager = FeatureExtractorManager(model=CNN_LSTM, encoder=self.encoder)
        >>> self.feature_extractor = self.feature_manager.get_feature_extractor()
        >>> self.feature_transformer = self.feature_manager.transform_features
        >>> self.feature_extractor_cnn = self.feature_manager.get_feature_extractor_cnn(
        ...     activation        = self.activation,
        ...     dropout           = self.dropout,
        ... )
        >>> self.lstm = nn.LSTM(
        ...         input_size=self.feature_manager.output_dim,
        ...         hidden_size=self.hidden_dim,
        ...         num_layers=self.num_layers,
        ...         batch_first=True,
        ...         dropout=dropout
        ... )
    """
    def __init__(
        self,
        model: str,
        encoder: str,
        output_dim: int = 1,
    ) -> None:
        self.model      = model
        self.encoder    = encoder or RAW
        self.output_dim = output_dim

    def get_feature_extractor(self) -> torch.nn.Module | None:
        """Returns feature extractor class."""
        return FEATURE_EXTRACTORS[self.encoder]

    def get_feature_extractor_cnn(
        self,
        activation: torch.nn.Module,
        dropout: torch.nn.Module,
    ) -> torch.nn.Sequential:
        """Returns layers of CNN and MaxPool according to the model and encoder types."""
        layers = []
        conv_count = 1
        conv2d_pms = CONV2D[self.encoder]
        max2d_pms = MAX2D[self.encoder]
        for layer in ORDER2D[self.model][self.encoder]:
            if layer == CONV:
                layers.append(self.__feature_extractor_conv2d(
                    conv2d_pms[f"in_{conv_count}"],
                    conv2d_pms[f"out_{conv_count}"],
                    conv2d_pms[f"kernel_{conv_count}"],
                    conv2d_pms[f"stride_{conv_count}"],
                    conv2d_pms[f"padding_{conv_count}"],
                    activation,
                    dropout,
                ))
                self.output_dim = conv2d_pms[f"out_{conv_count}"]
                conv_count += 1
            elif layer == MAXPOOL:
                layers.append(nn.MaxPool2d(
                    kernel_size = max2d_pms['kernel'],
                    stride      = max2d_pms['stride'],
                ))
        return nn.Sequential(*layers)

    def __feature_extractor_conv2d(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int],
        padding: int | tuple[int, int],
        activation: torch.nn.Module,
        dropout: torch.nn.Module,
    ) -> torch.nn.Sequential:
        """Returns the combination of CNN according to the model type."""
        if self.model == CNN_LSTM:
            return nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channel),
                activation,
                dropout,
            )
        elif self.model == RESNET_LSTM:
            return nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1),
                nn.BatchNorm2d(out_channel),
                activation,
            )

    def transform_features(self, x: torch.Tensor) -> torch.Tensor:
        """Transform features according to the encoder type."""
        if self.encoder == RAW:
            return x.unsqueeze(1)
        elif self.encoder == PSD:
            return x.reshape(x.size(0), -1, x.size(3)).unsqueeze(1)
        return x

