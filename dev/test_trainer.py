from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from pyhealth.evaluator.evaluating_multiclass import evaluate_multiclass
from pyhealth.trainer import Trainer


class MNISTDataset:
    def __init__(self, train=True):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.dataset = datasets.MNIST("../data", train=train, download=True, transform=transform)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return {"x": x, "y": y}

    def __len__(self):
        return len(self.dataset)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y, device, **kwargs):
        x = torch.stack(x, dim=0).to(device)
        y = torch.tensor(y).to(device)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        loss = self.loss(x, y)
        y_prob = F.softmax(x, dim=1)
        return {"loss": loss, "y_prob": y_prob, "y_true": y}


train_dataset = MNISTDataset(train=True)
eval_dataset = MNISTDataset(train=False)

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               collate_fn=lambda x: {k: [d[k] for d in x] for k in x[0]},
                                               batch_size=64,
                                               shuffle=True)
eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                              collate_fn=lambda x: {k: [d[k] for d in x] for k in x[0]},
                                              batch_size=64,
                                              shuffle=False)

model = Model()

trainer = Trainer(enable_logging=True, output_path="../output")
trainer.fit(model,
            train_loader=train_dataloader,
            epochs=5,
            evaluate_fn=evaluate_multiclass,
            eval_loader=eval_dataloader,
            monitor="acc")
