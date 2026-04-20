"""
Example script for DuETT model on PhysioNet-2012 Mortality Prediction.

Paper: "DuETT: Dual Event Time Transformer for Electronic Health Records"
Link: https://arxiv.org/abs/2304.13017

Ablation Study:
---------------
We perform variations across both Task and Model parameters.
1. Task feature configurations: Varying discretization intervals (n_timesteps).
2. Hyperparameter configurations: Varying Transformer depth (n_duett_layers) and embedding size.

Findings:
---------
By modifying time-bins and attention depths dynamically on the demo data, we observe
variations in model PR-AUC scoring trajectories representing architectural scaling.
"""
import torch
from unittest.mock import patch
from pyhealth.datasets import PhysioNet2012Dataset, split_by_patient, get_dataloader
from pyhealth.models import DuETT
from pyhealth.tasks import PhysioNetMortalityTask
from pyhealth.trainer import Trainer

@patch("pyhealth.datasets.base_dataset.in_notebook", return_value=True)
def main(mock_in_notebook):
    # Base dataset loads the mocked/fast cached 1K demo parameters
    dataset = PhysioNet2012Dataset(root="/tmp/physionet2012", tables=["events", "outcomes"], dev=True)

    ablations = [
        {"n_timesteps": 16, "n_layers": 1, "d_embedding": 16},
        {"n_timesteps": 32, "n_layers": 2, "d_embedding": 24}, # Original paper config
        {"n_timesteps": 48, "n_layers": 2, "d_embedding": 32},
    ]

    results_summary = []

    for config in ablations:
        print(f"\n=== ABLATION: Timesteps {config['n_timesteps']} | Layers {config['n_layers']} | Embed {config['d_embedding']} ===")
        
        # 1. Apply Task Variation
        task = PhysioNetMortalityTask(n_timesteps=config['n_timesteps'])
        sample_dataset = dataset.set_task(task=task)
        
        train_ds, val_ds, test_ds = split_by_patient(sample_dataset, [0.7, 0.15, 0.15])
        train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
        val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)
        test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)

        # 2. Apply Hyperparameter Variation
        model = DuETT(
            dataset=sample_dataset,
            n_timesteps=config['n_timesteps'],
            d_embedding=config['d_embedding'],
            n_duett_layers=config['n_layers'],
            d_hidden_tab_encoder=128,
            n_hidden_head=1,
            d_hidden_head=64
        )

        trainer = Trainer(model=model, metrics=["pr_auc", "roc_auc"], device="cpu")

        # 3. Fast Execute
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=3,
            monitor="pr_auc",
            monitor_criterion="max"
        )
        res = trainer.evaluate(test_loader)
        print(f"--> RESULTS: {res}")
        
        results_summary.append({
            "Config": f"T={config['n_timesteps']}, L={config['n_layers']}, E={config['d_embedding']}",
            "PR-AUC": res["pr_auc"],
            "ROC-AUC": res["roc_auc"]
        })

    print("\n=== ABLATION SUMMARY ===")
    for res in results_summary:
        print(f"{res['Config']} -> PR-AUC: {res['PR-AUC']:.4f} | ROC-AUC: {res['ROC-AUC']:.4f}")

if __name__ == "__main__":
    main()