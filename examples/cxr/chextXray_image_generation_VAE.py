"""Train a conditional VAE on the COVID-19 Radiography Database.

Download and extract the dataset from
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database,
then set ``DATA_ROOT`` to the extracted directory.
"""

import torch

from pyhealth.datasets import COVID19CXRDataset, get_dataloader
from pyhealth.models import VAE
from pyhealth.tasks import COVID19CXRClassification
from pyhealth.trainer import Trainer


DATA_ROOT = "/path/to/COVID-19_Radiography_Dataset"
BATCH_SIZE = 32
EPOCHS = 10

base_dataset = COVID19CXRDataset(root=DATA_ROOT)
dataset = base_dataset.set_task(COVID19CXRClassification())
train_dataloader = get_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = VAE(
    dataset=dataset,
    feature_key="image",
    condition_key="disease",
    latent_dim=128,
    hidden_dim=32,
)
trainer = Trainer(
    model=model,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
trainer.train(train_dataloader=train_dataloader, epochs=EPOCHS)

# Sample one image for each class in the fitted disease vocabulary.
label_vocab = dataset.output_processors["disease"].label_vocab
conditions = torch.arange(len(label_vocab))
generated_images = trainer.model.generate(conditions=conditions)
print("Label indices:", label_vocab)
print("Generated images:", generated_images.shape)
