from torchvision import transforms

from pyhealth.datasets import COVID19XRayDataset, get_dataloader
from pyhealth.datasets import split_by_patient
from pyhealth.models import ResNet
from pyhealth.tasks import covid19_classification_fn
from pyhealth.trainer import Trainer

# STEP 1: load data
base_dataset = COVID19XRayDataset(
    root="/srv/local/data/zw12/raw_data/covid19-radiography-database/COVID-19_Radiography_Dataset",
)
base_dataset.stat()

# STEP 2: set task
sample_dataset = base_dataset.set_task(covid19_classification_fn)
sample_dataset.set_transform(transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5862785803043838], std=[0.27950088968644304])
]))

train_dataset, val_dataset, test_dataset = split_by_patient(
    sample_dataset, [0.8, 0.1, 0.1]
)
train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

# STEP 3: define model
model = ResNet(
    dataset=sample_dataset,
    feature_keys=[
        "image",
    ],
    label_key="label",
    mode="multiclass",
    num_layers=18,
)

# STEP 4: define trainer
trainer = Trainer(model=model)
trainer.train(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=3,
    monitor="accuracy",
)

# STEP 5: evaluate
print(trainer.evaluate(test_dataloader))
