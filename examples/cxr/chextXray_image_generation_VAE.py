"""Train a conditional VAE on the COVID-19 Radiography Database.

Download and extract the dataset from
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database,
then pass its directory to :func:`main` or set ``DATA_ROOT`` below.
"""

import matplotlib.pyplot as plt
import torch

from pyhealth.datasets import (
    COVID19CXRDataset,
    get_dataloader,
    split_by_sample,
)
from pyhealth.models import VAE
from pyhealth.tasks import COVID19CXRClassification
from pyhealth.trainer import Trainer


DATA_ROOT = "/path/to/COVID-19_Radiography_Dataset"
BATCH_SIZE = 32
EPOCHS = 10


def main(root: str = DATA_ROOT) -> None:
    """Train, evaluate, and sample from the conditional image VAE."""
    base_dataset = COVID19CXRDataset(root=root)
    dataset = base_dataset.set_task(COVID19CXRClassification())
    train_dataset, val_dataset, test_dataset = split_by_sample(
        dataset, [0.8, 0.1, 0.1], seed=0
    )
    train_loader = get_dataloader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = get_dataloader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = get_dataloader(test_dataset, batch_size=BATCH_SIZE)

    model = VAE(
        dataset=train_dataset,
        feature_key="image",
        condition_key="disease",
        hidden_dim=128,
    )
    trainer = Trainer(
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        metrics=["mse", "mae"],
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=EPOCHS,
        monitor="mse",
        monitor_criterion="min",
    )

    print("Test reconstruction metrics:", trainer.evaluate(test_loader))
    y_true, y_prob, _ = trainer.inference(test_loader)
    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(y_true[0, 0], cmap="gray")
    axes[0].set_title("Input image")
    axes[1].imshow(y_prob[0, 0], cmap="gray")
    axes[1].set_title("Reconstruction")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    plt.show()

    # Sample one image for each class in the fitted disease vocabulary.
    label_vocab = dataset.output_processors["disease"].label_vocab
    conditions = torch.arange(len(label_vocab))
    generated_images = trainer.model.generate(conditions=conditions).cpu()
    print("Label indices:", label_vocab)
    print("Generated images:", generated_images.shape)


if __name__ == "__main__":
    main()
