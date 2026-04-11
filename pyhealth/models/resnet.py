"""Plain 1-D ResNet-18 ECG model.

Implements the ``resnet1d18`` backbone used in:

    Nonaka N. & Seita J. (2021). In-depth Benchmarking of Deep Neural Network
    Architectures for ECG Diagnosis. *PMLR* 149:1–19.
    https://proceedings.mlr.press/v149/nonaka21a.html

See :mod:`pyhealth.models.resnet_ecg_base` for the shared building blocks.
"""

from pyhealth.datasets import SampleDataset
from pyhealth.models.resnet_ecg_base import BasicBlock1d, ECGBackboneModel, ResNet1d


class ResNet18ECG(ECGBackboneModel):
    """ResNet-18 backbone for ECG classification (Nonaka & Seita, 2021).

    Standard 1-D ResNet-18 with a two-layer prediction head.

    **Backbone** (``resnet1d18`` in the reference code):

    * Stem: ``Conv1d(12, 64, 7, stride=2) → BN → ReLU → MaxPool1d(3, stride=2)``
    * Four stages of :class:`~pyhealth.models.BasicBlock1d` with layer counts
      ``[2, 2, 2, 2]`` and channel widths ``[64, 128, 256, 512]``.
    * ``AdaptiveAvgPool1d(1)`` → ``Linear(512, backbone_output_dim)``.

    **Head** (``HeadModule`` in the reference code):

    ``Linear(256, 128) → ReLU → BN(128) → Dropout(0.25) → Linear(128, n_classes)``

    **Windowing** (Section 4.2):

    During *training* a random 2.5-second window is cropped from each
    recording (handle in the task preprocessing / collate function).
    During *evaluation* use :meth:`forward_sliding_window` for the 50 %-overlap
    sliding-window protocol from the paper.

    Args:
        dataset (SampleDataset): Dataset used to infer feature/label keys,
            output size, and loss function.
        in_channels (int): Number of ECG leads. Default ``12``.
        base_channels (int): Width of the first residual stage. Default ``64``.
        backbone_output_dim (int): Backbone projection output dimension.
            Default ``256``.
        dropout (float): Dropout probability in the prediction head.
            Default ``0.25``.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {"patient_id": "p0", "visit_id": "v0",
        ...      "signal": np.random.randn(12, 1250).astype(np.float32),
        ...      "label": [1, 0, 1, 0, 0]},
        ...     {"patient_id": "p1", "visit_id": "v1",
        ...      "signal": np.random.randn(12, 1250).astype(np.float32),
        ...      "label": [0, 1, 0, 1, 0]},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"signal": "tensor"},
        ...     output_schema={"label": "multilabel"},
        ...     dataset_name="test",
        ... )
        >>> model = ResNet18ECG(dataset=dataset)
        >>> out = model(**next(iter(get_dataloader(dataset, batch_size=2))))
        >>> sorted(out.keys())
        ['logit', 'loss', 'y_prob', 'y_true']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        in_channels: int = 12,
        base_channels: int = 64,
        backbone_output_dim: int = 256,
        dropout: float = 0.25,
    ) -> None:
        super().__init__(dataset=dataset)

        self.backbone = ResNet1d(
            in_channels=in_channels,
            layers=[2, 2, 2, 2],
            block=BasicBlock1d,
            base_channels=base_channels,
            output_dim=backbone_output_dim,
        )
        self._build_head(backbone_output_dim, dropout)
