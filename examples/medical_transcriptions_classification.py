import numpy as np
import torch

from pyhealth.datasets import MedicalTranscriptionsDataset
from pyhealth.datasets import get_dataloader
from pyhealth.models import HuggingfaceAutoModel
from pyhealth.trainer import Trainer

root = "/srv/local/data/zw12/raw_data/MedicalTranscriptions"
base_dataset = MedicalTranscriptionsDataset(root)

sample_dataset = base_dataset.set_task()

ratios = [0.7, 0.1, 0.2]
index = np.arange(len(sample_dataset))
np.random.shuffle(index)
s1 = int(len(sample_dataset) * ratios[0])
s2 = int(len(sample_dataset) * (ratios[0] + ratios[1]))
train_index = index[: s1]
val_index = index[s1: s2]
test_index = index[s2:]
train_dataset = torch.utils.data.Subset(sample_dataset, train_index)
val_dataset = torch.utils.data.Subset(sample_dataset, val_index)
test_dataset = torch.utils.data.Subset(sample_dataset, test_index)

train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

model = HuggingfaceAutoModel(
    model_name="emilyalsentzer/Bio_ClinicalBERT",
    dataset=sample_dataset,
    feature_keys=["transcription"],
    label_key="label",
    mode="multiclass",
)

trainer = Trainer(model=model)

print(trainer.evaluate(test_dataloader))

trainer.train(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=1,
    monitor="accuracy"
)

print(trainer.evaluate(test_dataloader))
