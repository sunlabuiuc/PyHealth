import torch
from typing import Dict
from torch.utils.data import IterableDataset
from pyhealth.sampler import TUSZSampler


class TUSZSamplerDataset(IterableDataset):
    def __init__(
            self,
            dataset,
            is_training_set,
            buffer_size = 1000
    ):
        self.dataset = dataset
        self.is_training_set = is_training_set
        self.buffer_size = buffer_size

    task_name: str = "tusz_sampler_task"
    input_schema: Dict[str, str] = { "signal": "tensor" }
    output_schema: Dict[str, str] = {
        "label": "tensor",
        "label_bitgt_1": "tensor",
        "label_bitgt_2": "tensor",
        "label_name": "text",
    }

        
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
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
