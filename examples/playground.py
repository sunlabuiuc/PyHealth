from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from pyhealth.data.dataset import DrugRecDataLoader
from pyhealth.datasets import MIMIC3BaseDataset
from pyhealth.models import RETAIN
from pyhealth.evaluator import DrugRecommendationEvaluator

# dataset
base_dataset = MIMIC3BaseDataset(root="/srv/local/data/physionet.org/files/mimiciii/1.4")
# task
task_taskset = DrugRecDataLoader(base_dataset)
# model
model = RETAIN(task_taskset)
# trainer
trainer = Trainer(
    gpus=1,
    max_epochs=3,
    progress_bar_refresh_rate=5,
)
# evaluator
evaluator = DrugRecommendationEvaluator(model)
# training
trainer.fit(
    model=model,
    train_dataloaders=DataLoader(task_taskset, batch_size=1, collate_fn=lambda x: x[0]),
)
# evaluating
# evaluator.evaluate()
