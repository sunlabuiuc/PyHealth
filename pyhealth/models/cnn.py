from typing import List, Tuple, Union, Dict

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torchvision.models import resnext50_32x4d
import torchvision.transforms as transforms

from pyhealth.datasets import BaseDataset
from pyhealth.models import BaseModel
from pyhealth.tokenizer import Tokenizer
from pyhealth.models.utils import get_default_loss_module

import numpy as np

from pyhealth.models import ClassicML


class CNN(BaseModel):
    """CNN Class"""

    def __init__(
        self,
        dataset: BaseDataset,
        tables: Union[List[str], Tuple[str]],
        target: str,
        mode: str,
        model: nn.Module = None,
        **kwargs
    ):
        super(CNN, self).__init__(
            dataset=dataset,
            tables=tables,
            target=target,
            mode=mode,
        )

        self.tables = tables
        self.target = target
        self.mode = mode
        self.tokenizers = {}
        for domain in tables:
            self.tokenizers[domain] = Tokenizer(
                dataset.get_all_tokens(key=domain), special_tokens=["<pad>", "<unk>"]
            )
        self.label_tokenizer = Tokenizer(dataset.get_all_tokens(key=target))

        self.patients = dataset.parse_tables()

        # determine the height of the transferred image
        self.max_visits = 0
        for key in self.patients.keys():
            visit_times = len(self.patients[key].visits)
            if visit_times > self.max_visits:
                self.max_visits = visit_times

        # Default CNN model
        if model is None:
            model = resnext50_32x4d(pretrained=False)

        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=model.fc.in_features,
                out_features=self.label_tokenizer.get_vocabulary_size()
            ),
        )
        self.model = model
        self.sigmoid = nn.Sigmoid()

    def forward(self, device, **kwargs):
        # After transfer code to image, the real batch size for each batch would be different
        # So, we re-define the input batch as the input to the CNN model
        cur_X, cur_y = ClassicML.code2vec(self, **kwargs)
        X, y = code2image(cur_X=cur_X,
                          cur_y=cur_y,
                          batch=kwargs,
                          label_tokenizer=self.label_tokenizer,
                          max_visits=self.max_visits,
                          )
        data = CNNTaskData(X, y)
        dataloader = DataLoader(data, batch_size=8, shuffle=False)

        logit = None
        y_true = None

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = self.model(inputs)
            if logit is None:
                logit = outputs
            else:
                logit = torch.cat([logit, outputs], dim=0)
            if y_true is None:
                y_true = targets
            else:
                y_true = torch.cat([y_true, targets], dim=0)

        loss = get_default_loss_module(self.mode)(logit, y_true.float())
        y_prob = torch.sigmoid(logit)
        y_pred = (y_prob > 0.5).int()

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_pred": y_pred,
            "y_true": y_true,
        }


def code2image(
        cur_X: np.array,
        cur_y: np.array,
        batch: dict,
        label_tokenizer: Tokenizer,
        max_visits: int,
        entity: str = "patient_id",
        resize: bool = False,
        width: int = 512
):
    """
    Transfer codes in a batch to entity-wise images (an entity is a patient in usual)
    max_visits: maximum visits of an entity in the whole dataset, is used as the height of the image
    resize: a boolean value to show whether to resize the image
    width: a parameter to specify the width of the image if resize=True
    """
    X = []
    y = []

    X_dict_by_entity = {}
    y_dict_by_entity = {}
    for eid in set(batch[entity]):
        X_dict_by_entity[eid] = []
        y_dict_by_entity[eid] = np.zeros(label_tokenizer.get_vocabulary_size())

    for i in range(len(batch[entity])):
        X_dict_by_entity[batch[entity][i]].append(cur_X[i])
        y_dict_by_entity[batch[entity][i]] += cur_y[i]

    if resize:
        transform = transforms.Compose(
            [
                transforms.Resize((max_visits, width)),
                transforms.Normalize(cur_X.mean(), cur_X.std())
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((max_visits, len(cur_X[0]))),
                transforms.Normalize(cur_X.mean(), cur_X.std())
            ]
        )

    # transfer the 2d-array of concatenated domain tokens of each entity to tensor (an image)
    # and make a multi-hot label for each entity
    for eid in set(batch[entity]):
        x = np.array(X_dict_by_entity[eid], dtype=int)
        l = []
        for i in range(3):
            l.append(x)
        l = np.array(l)
        x = transform(torch.from_numpy(l).float())
        X.append(x)
        y_ = (y_dict_by_entity[eid] > 0).astype(int)
        y.append(y_)

    return X, y


class CNNTaskData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

