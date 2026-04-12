import torch
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
