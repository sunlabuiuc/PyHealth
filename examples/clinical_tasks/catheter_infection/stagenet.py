import argparse

import torch

from pyhealth.datasets import MIMIC4Dataset, get_dataloader, split_by_patient
from pyhealth.models import StageNet
from pyhealth.tasks import CatheterAssociatedInfectionPredictionStageNetMIMIC4
from pyhealth.trainer import Trainer

MIMIC4_ROOT = "/shared/rsaas/physionet.org/files/mimiciv/2.2"
CACHE_DIR = "/shared/eng/pyhealth_agent/cache"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--padding", type=int, default=10)
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    base_dataset = MIMIC4Dataset(
        ehr_root=MIMIC4_ROOT,
        cache_dir=CACHE_DIR,
        ehr_tables=[
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "prescriptions",
            "labevents",
        ],
        num_workers=4,
    )

    sample_dataset = base_dataset.set_task(
        CatheterAssociatedInfectionPredictionStageNetMIMIC4(padding=args.padding),
        num_workers=4,
    )
    print(f"Total samples: {len(sample_dataset)}")

    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset,
        [0.8, 0.1, 0.1],
    )

    train_loader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    model = StageNet(
        dataset=sample_dataset,
        embedding_dim=128,
        chunk_size=128,
        levels=3,
        dropout=0.3,
    )

    trainer = Trainer(
        model=model,
        device=device,
        metrics=["pr_auc", "roc_auc", "precision", "f1", "accuracy"],
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        monitor="precision",
        optimizer_params={"lr": 1e-4},
    )

    results = trainer.evaluate(test_loader)
    print("\nTest Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
