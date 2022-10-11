from typing import List, Tuple, Union, Dict

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from pyhealth.data import BaseDataset
from pyhealth.models import BaseModel
from pyhealth.tokenizer import Tokenizer

import numpy as np

from MLModel import code2vec



class CNN(BaseModel):
    """CNN Class"""

    def __init__(
        self,
        dataset: BaseDataset,
        tables: Union[List[str], Tuple[str]],
        target: str,
        mode: str,
        image_size: Tuple = (512, 512),
        model: str = "resnet",
        **kwargs
    ):
        super(CNN, self).__init__(
            dataset=dataset,
            tables=tables,
            target=target,
            mode=mode,
        )

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


        if model == "resnet":
            resnet = torchvision.models.resnext50_32x4d(pretrained=False)
            resnet.fc = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(
                    in_features=resnet.fc.in_features, out_features=n_classes
                ),
            )
            self.model = resnet
            self.sigmoid = nn.Sigmoid()

    def forward(self, device, **kwargs):
            return self.sigmoid(self.model(**kwargs))


def code2image(
        tables: Union[List[str], Tuple[str]],
        target: str,
        domain_tokenizers: Dict[str, Tokenizer],
        label_tokenizer: Tokenizer,
        batch: dict,
        max_visits: int,
        entity: str = "patient_id",
        resize: bool = False,
        width: int = 512
):
    X = []
    y = []

    cur_X, cur_y = code2vec(tables, target, domain_tokenizers, label_tokenizer, batch)

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




