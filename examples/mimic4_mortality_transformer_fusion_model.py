"""Example of using TransformerFusionModel for MIMIC-IV mortality prediction.

This script shows how to initialize the model, compare hyperparameter choices,
and run a quick train/evaluate workflow using MIMIC-IV mortality prediction.
"""

import os

from pyhealth.datasets import MIMIC4Dataset, get_dataloader, split_by_patient
from pyhealth.models import TransformerFusionModel
from pyhealth.tasks import MortalityPredictionMIMIC4
from pyhealth.trainer import Trainer


def build_model(dataset, embedding_dim, num_heads, num_layers, dropout):
    return TransformerFusionModel(
        dataset=dataset,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        use_modality_token=False,
    )


def main():
    data_root = "/srv/local/data/physionet.org/files/mimiciv/2.2/"
    if not os.path.exists(data_root):
        raise FileNotFoundError(
            f"MIMIC-IV root does not exist: {data_root}. Update the path before running."
        )

    base_dataset = MIMIC4Dataset(
        ehr_root=data_root,
        ehr_tables=[
            "patients",
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "labevents",
        ],
        num_workers=8,
    )

    task = MortalityPredictionMIMIC4(padding=20)
    sample_dataset = base_dataset.set_task(task, num_workers=8)

    print("Input schema:", sample_dataset.input_schema)
    print("Output schema:", sample_dataset.output_schema)

    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )

    train_loader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    configs = [
        {"embedding_dim": 64, "num_heads": 4, "num_layers": 1, "dropout": 0.1},
        {"embedding_dim": 128, "num_heads": 4, "num_layers": 2, "dropout": 0.2},
    ]

    for config in configs:
        print("\n=== Running config ===")
        print(config)

        model = build_model(
            dataset=sample_dataset,
            embedding_dim=config["embedding_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        )

        trainer = Trainer(
            model=model,
            device="cpu",
            metrics=["roc_auc", "accuracy"],
            output_path="./output/transformer_fusion_model",
        )

        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=1,
            monitor="roc_auc",
            optimizer_params={"lr": 1e-4},
        )

        results = trainer.evaluate(test_loader)
        print("Results:", results)


if __name__ == "__main__":
    main()
