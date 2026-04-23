# Author: Wang, Ziquan ziquanw2@illinois.edu
import os

import torch

from pyhealth.datasets import DermoscopyDataset, get_dataloader, split_by_sample
from pyhealth.models import TorchvisionModel
from pyhealth.processors import DermoscopyImageProcessor
from pyhealth.tasks import DermoscopyMelanomaClassification
from pyhealth.trainer import Trainer

# Root of the dermoscopy_data folder (see README for the expected layout).
DATA_ROOT = "/Users/ziquanwang/Documents/UIUC MCS/CS 598 DLH/Project/dermoscopy_data"
MODE = "whole"  # one of: whole, lesion, background, bbox, bbox70, bbox90,
#         high_whole, high_lesion, high_background,
#         low_whole, low_lesion, low_background
BATCH_SIZE = 32
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Since PyHealth uses multiprocessing, it is best practice to use a main guard.
if __name__ == "__main__":
    dataset = DermoscopyDataset(
        root=DATA_ROOT,
        cache_dir=os.path.join(DATA_ROOT, "cache"),
    )
    dataset.stats()

    task = DermoscopyMelanomaClassification("ph2")
    processor = DermoscopyImageProcessor(mode=MODE, image_size=224, normalize=True)
    samples = dataset.set_task(task=task, input_processors={"image": processor})

    train_dataset, val_dataset, test_dataset = split_by_sample(samples, [0.7, 0.1, 0.2])

    train_loader = get_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TorchvisionModel(
        dataset=samples,
        model_name="resnet50",
        model_config={"weights": "DEFAULT"},
    )

    trainer = Trainer(model=model, device=DEVICE)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=NUM_EPOCHS,
        optimizer_params={"lr": LEARNING_RATE},
        monitor="roc_auc",
    )

    trainer.evaluate(test_loader)

    samples.close()
