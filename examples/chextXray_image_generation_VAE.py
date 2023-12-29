from pyhealth.datasets import split_by_visit, get_dataloader
from pyhealth.trainer import Trainer
from pyhealth.datasets import COVID19CXRDataset
from pyhealth.models import VAE
from torchvision import transforms

import torch
import numpy as np

# step 1: load signal data
root = "/srv/local/data/COVID-19_Radiography_Dataset"
base_dataset = COVID19CXRDataset(root)

# step 2: set task
sample_dataset = base_dataset.set_task()

# the transformation automatically normalize the pixel intensity into [0, 1]
transform = transforms.Compose([
    transforms.Lambda(lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1)), # only use the first channel
    transforms.Resize((128, 128)),
])

def encode(sample):
    sample["path"] = transform(sample["path"])
    return sample

sample_dataset.set_transform(encode)


# split dataset
train_dataset, val_dataset, test_dataset = split_by_visit(
    sample_dataset, [0.6, 0.2, 0.2]
)
train_dataloader = get_dataloader(train_dataset, batch_size=256, shuffle=True)
val_dataloader = get_dataloader(val_dataset, batch_size=256, shuffle=False)
test_dataloader = get_dataloader(test_dataset, batch_size=256, shuffle=False)

data = next(iter(train_dataloader))
print (data)

print (data["path"][0].shape)

print(
    "loader size: train/val/test",
    len(train_dataset),
    len(val_dataset),
    len(test_dataset),
)

# STEP 3: define model
model = VAE(
    dataset=sample_dataset,
    input_channel=3,
    input_size=128,
    feature_keys=["path"],
    label_key="path",
    mode="regression",
    hidden_dim = 128,
)

# STEP 4: define trainer
trainer = Trainer(model=model, device="cuda:4", metrics=["kl_divergence", "mse", "mae"])
trainer.train(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=10,
    monitor="kl_divergence",
    monitor_criterion="min",
    optimizer_params={"lr": 1e-3},
)

# # STEP 5: evaluate
# print(trainer.evaluate(test_dataloader))


import matplotlib.pyplot as plt

# EXP 1: check the real chestxray image and the reconstructed image
X, X_rec, _ = trainer.inference(test_dataloader)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(X[0].reshape(128, 128), cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(X_rec[0].reshape(128, 128), cmap="gray")
plt.savefig("chestxray_vae_comparison.png")

# EXP 2: random images
model = trainer.model
  
model.eval()
with torch.no_grad():
    x = np.random.normal(0, 1, 128)
    x = x.astype(np.float32)
    x = torch.from_numpy(x).to(trainer.device)
    rec = model.decoder(x).detach().cpu().numpy()
    rec = rec.reshape((128, 128))
    plt.figure()
    plt.imshow(rec, cmap="gray")
    plt.savefig("chestxray_vae_synthetic.png")