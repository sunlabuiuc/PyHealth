import torch
from torch.utils.data import DataLoader

from pyhealth.split import split_by_patient
from pyhealth.datasets import MIMIC3BaseDataset
from pyhealth.evaluator.evaluating_multilabel import evaluate_multilabel
from pyhealth.models.rnn import RNNModel
from pyhealth.tasks import DrugRecommendationDataset
from pyhealth.trainer import Trainer
from pyhealth.utils import collate_fn_dict

base_dataset = MIMIC3BaseDataset(root="/srv/local/data/physionet.org/files/mimiciii/1.4", dev=True)
task_dataset = DrugRecommendationDataset(base_dataset)

train_dataset, val_dataset, test_dataset = split_by_patient(task_dataset, [0.8, 0.1, 0.1])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_dict)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn_dict)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn_dict)

model = RNNModel(dataset=task_dataset,
                 input_domains=["conditions", "procedures"],
                 output_domain="drugs",
                 mode="multilabel")

trainer = Trainer(enable_logging=True, output_path="../output")
trainer.fit(model,
            train_loader=train_loader,
            epochs=50,
            evaluate_fn=evaluate_multilabel,
            eval_loader=val_loader,
            monitor="jaccard")

print("Evaluate on test set")
