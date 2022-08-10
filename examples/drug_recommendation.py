from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from pyhealth.data.split import random_split
from pyhealth.datasets import MIMIC3BaseDataset
from pyhealth.evaluator import DrugRecommendationEvaluator
from pyhealth.models import RNN
from pyhealth.tasks import DrugRecommendationDataset

# read raw dataset
base_dataset = MIMIC3BaseDataset(root="/srv/local/data/physionet.org/files/mimiciii/1.4")

# convert dataset for drug recommendation task
task_dataset = DrugRecommendationDataset(base_dataset)

# split dataset into train, val, and test
train_dataset, val_dataset, test_dataset = random_split(task_dataset, ratios=[0.7, 0.1, 0.2])

# create dataloader
train_data_loader = DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x[0])
val_data_loader = DataLoader(val_dataset, batch_size=1, collate_fn=lambda x: x[0])
test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=lambda x: x[0])

# build model
model = RNN(task_dataset)

# trainer
trainer = Trainer(
    gpus=1,
    max_epochs=1,
    progress_bar_refresh_rate=5,
)

# evaluator
evaluator = DrugRecommendationEvaluator(model=model)

# train
trainer.fit(
    model=model,
    train_dataloaders=train_data_loader,
    val_dataloaders=val_data_loader,
)

# evaluate
evaluator.evaluate(test_data_loader, device="cuda")
