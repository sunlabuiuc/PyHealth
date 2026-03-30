"""
Optuna hyperparameter tuning for RNN on drug recommendation with MIMIC-IV.

This example demonstrates:
1. Loading MIMIC-IV data and applying the DrugRecommendationMIMIC4 task
2. Defining an Optuna objective that tunes RNN-specific hyperparameters
3. Running 10 Optuna trials to find the best configuration
4. Training a final model with the best hyperparameters

Tuned hyperparameters:
    - embedding_dim: embedding size for code tokens
    - hidden_dim: GRU/LSTM/RNN hidden state size
    - rnn_type: recurrent cell type (GRU, LSTM, RNN)
    - num_layers: number of stacked recurrent layers
    - dropout: dropout rate applied before each recurrent layer
    - lr: learning rate for AdamW
    - weight_decay: L2 regularization coefficient for AdamW
"""

import torch
import optuna

from pyhealth.datasets import (
    MIMIC4Dataset,
    get_dataloader,
    split_by_patient,
)
from pyhealth.models import RNN
from pyhealth.tasks import DrugRecommendationMIMIC4
from pyhealth.trainer import Trainer

if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # STEP 1: Load MIMIC-IV base dataset
    # ---------------------------------------------------------------------------
    base_dataset = MIMIC4Dataset(
        ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.2/",
        cache_dir="/shared/eng/pyhealth_agent/baselines",
        ehr_tables=[
            "patients",
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "prescriptions",
        ],
    )

    # STEP 2: Apply drug recommendation task
    sample_dataset = base_dataset.set_task(
        DrugRecommendationMIMIC4(),
        num_workers=4,
    )

    print(f"Total samples: {len(sample_dataset)}")
    print(f"Input schema:  {sample_dataset.input_schema}")
    print(f"Output schema: {sample_dataset.output_schema}")

    # STEP 3: Split dataset (fixed split so all trials see the same data)
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )

    print(f"\nDataset split — Train: {len(train_dataset)}  "
          f"Val: {len(val_dataset)}  Test: {len(test_dataset)}")

    # ---------------------------------------------------------------------------
    # STEP 4: Define Optuna objective
    # ---------------------------------------------------------------------------
    DEVICE = "cuda:2"   # or "cpu"
    TUNE_EPOCHS = 10    # lightweight training per trial
    N_TRIALS = 10

    def objective(trial: optuna.Trial) -> float:
        """Return validation pr_auc_samples for a sampled RNN configuration."""

        # --- Suggest hyperparameters -------------------------------------------
        embedding_dim = trial.suggest_categorical(
            "embedding_dim", [64, 128, 256]
        )
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        rnn_type = trial.suggest_categorical(
            "rnn_type", ["GRU", "LSTM", "RNN"]
        )
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.7)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        # --- Build dataloaders -------------------------------------------------
        train_loader = get_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = get_dataloader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        # --- Build model -------------------------------------------------------
        model = RNN(
            dataset=sample_dataset,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            rnn_type=rnn_type,
            num_layers=num_layers,
            dropout=dropout,
        )

        # --- Train -------------------------------------------------------------
        trainer = Trainer(
            model=model,
            device=DEVICE,
            metrics=["pr_auc_samples"],
        )
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=TUNE_EPOCHS,
            monitor="pr_auc_samples",
            optimizer_class=torch.optim.AdamW,
            optimizer_params={"lr": lr},
            weight_decay=weight_decay,
        )

        # --- Evaluate on validation set ----------------------------------------
        scores = trainer.evaluate(val_loader)
        return scores["pr_auc_samples"]

    # ---------------------------------------------------------------------------
    # STEP 5: Run Optuna study
    # ---------------------------------------------------------------------------
    print(
        f"\nStarting Optuna search ({N_TRIALS} trials, {TUNE_EPOCHS} epochs each)..."
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    best_params = study.best_params
    print("\nBest hyperparameters found:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"Best validation pr_auc_samples: {study.best_value:.4f}")

    # ---------------------------------------------------------------------------
    # STEP 6: Train final model with best hyperparameters
    # ---------------------------------------------------------------------------
    print("\nTraining final model with best hyperparameters...")

    train_loader = get_dataloader(
        train_dataset, batch_size=best_params["batch_size"], shuffle=True
    )
    val_loader = get_dataloader(
        val_dataset, batch_size=best_params["batch_size"], shuffle=False
    )
    test_loader = get_dataloader(
        test_dataset, batch_size=best_params["batch_size"], shuffle=False
    )

    final_model = RNN(
        dataset=sample_dataset,
        embedding_dim=best_params["embedding_dim"],
        hidden_dim=best_params["hidden_dim"],
        rnn_type=best_params["rnn_type"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
    )

    num_params = sum(p.numel() for p in final_model.parameters())
    print(f"Final model: {num_params:,} parameters")

    final_trainer = Trainer(
        model=final_model,
        device=DEVICE,
        metrics=["pr_auc_samples", "f1_samples", "jaccard_samples"],
    )
    final_trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=50,
        monitor="pr_auc_samples",
        optimizer_class=torch.optim.AdamW,
        optimizer_params={"lr": best_params["lr"]},
        weight_decay=best_params["weight_decay"],
    )

    # STEP 7: Evaluate on test set
    print("\nEvaluating on test set...")
    results = final_trainer.evaluate(test_loader)
    print("\nTest Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    # STEP 8: Inspect model predictions
    print("\nSample predictions:")
    sample_batch = next(iter(test_loader))

    with torch.no_grad():
        output = final_model(**sample_batch)

    print(f"  Batch size: {output['y_prob'].shape[0]}")
    print(f"  Number of drug classes: {output['y_prob'].shape[1]}")
    print("  Predicted probabilities (first 5 drugs of first patient):")
    print(f"    {output['y_prob'][0, :5].cpu().numpy()}")
    print("  True labels (first 5 drugs of first patient):")
    print(f"    {output['y_true'][0, :5].cpu().numpy()}")
