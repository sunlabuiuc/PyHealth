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
from typing import Dict, Optional
from torch.utils.data import IterableDataset
from pyhealth.sampler import TUSZSampler
from pyhealth.datasets import SampleDataset


class TUSZSamplerDataset(IterableDataset):
    """Wrapper dataset for WeightedRandomSampler the TUH Seizure Corpus (TUSZ)

    Dataset is available at https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml.
    
    Args:
        dataset        : an instance of BaseDataset.
        is_training_set: is dataset a training set.
        buffer_size    : buffer size from which the sampler chooses from.

    Examples:
        >>> from pyhealth.datasets import TUSZDataset
        >>> from pyhealth.tasks import TUSZTask
        >>> dataset = TUSZDataset(root = 'tuh_eeg_v2.0.5', subset = 'train')
        >>> task = TUSZTask(
        ...     sample_rate = SAMPLE_RATE,
        ...     feature_sample_rate = FEATURE_SAMPLE_RATE,
        ... )
        >>> samples = dataset.set_task(task)
        >>> from pyhealth.datasets import TUSZSamplerDataset
        >>> from torch.utils.data import DataLoader
        >>> sampler_ds = TUSZSamplerDataset(
        ...     dataset=dataset,
        ...     is_training_set=True,
        ...     buffer_size=32
        ... )
        >>> train_loader = DataLoader(
        ...     sampler_ds,
        ...     batch_size=BATCH_SIZE,
        ...     num_workers=1,
        ...     collate_fn=eeg_collate_fn,
        ...     drop_last=True,
        ...     shuffle = False,
        ... )
        >>> from pyhealth.models import CNNLSTM, ResNetLSTM
        >>> model = CNNLSTM(
        ...     dataset    = sampler_ds,
        ...     encoder    = 'raw',
        ...     num_layers = 1,
        ...     output_dim = OUTPUT_DIM,
        ...     batch_size = BATCH_SIZE,
        ...     dropout    = 0.5,
        ... )
    """

    task_name: str = "tusz_sampler_task"
    input_schema: Dict[str, str] = { "signal": "tensor" }
    output_schema: Dict[str, str] = {
        "label": "tensor",
        "label_bitgt_1": "tensor",
        "label_bitgt_2": "tensor",
        "label_name": "text",
    }

    def __init__(
            self,
            dataset: SampleDataset,
            is_training_set: Optional[bool] = True,
            buffer_size: Optional[int] = 1000,
    ) -> None:
        self.dataset = dataset
        self.is_training_set = is_training_set
        self.buffer_size = buffer_size
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __iter__(self):
        """Selects samples with WeightedRandomSampler"""
        buffer = []
        for data in self.dataset:
            buffer.append(data)
            if len(buffer) >= self.buffer_size:
                weights = TUSZSampler(
                    dataset=buffer,
                    is_training_set=self.is_training_set
                ).get_weights()
                indices = torch.multinomial(weights, len(weights), replacement=True).tolist()
                for idx in indices:
                    yield buffer[idx]
                buffer = []

        for data in buffer:
            yield data
