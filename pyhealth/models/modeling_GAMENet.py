from gettext import npgettext
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pyhealth.models.tokenizer import Tokenizer


if __name__ == "__main__":
    from pyhealth.datasets.mimic3 import MIMIC3BaseDataset
    from pyhealth.data.dataset import DrugRecommendationDataset
    from torch.utils.data import DataLoader

    base_dataset = MIMIC3BaseDataset(
        root="/srv/local/data/physionet.org/files/mimiciii/1.4"
    )
    task_taskset = DrugRecommendationDataset(base_dataset)
    data_loader = DataLoader(task_taskset, batch_size=1, collate_fn=lambda x: x[0])
    data_loader_iter = iter(data_loader)
    batch = next(data_loader_iter)
    model = RETAIN(task_taskset)
    print(model.training_step(batch, 0))
