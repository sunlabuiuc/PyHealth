"""PyHealth 2.0 training pipeline for shallow EEG-GAT.

10-fold cross-validation using PyHealth 2.0 Trainer, EEGGCNNDataset, and
the shallow EEGGATConvNet (2 GAT layers).  Dropout and attention dropout
ablations are supported via DROPOUT_VALUES and ATTN_DROPOUT_VALUES — one
full 10-fold CV run is executed per (dropout, attn_dropout) combination.

Dataset loading:
  EEGGCNNDataset.set_task() is called once to build and cache a SampleDataset
  on disk.  Subject-level 70/30 and fold splits are then made by indexing into
  SampleDataset.patient_to_index — no data is re-loaded between folds.

Checkpoints are saved as pure state-dicts:
    {EXPERIMENT_NAME}_drop{dropout*10}_attn{attn_dropout*10}_fold_{fold_idx}.ckpt

Load with:
    model.load_state_dict(torch.load("psd_gat_shallow_ph_drop2_attn0_fold_0.ckpt"))

Usage (from the examples/eeg_gatcnn directory):
    conda activate pyhealth (assuming PyHealth is installed in this conda env)
    python eeg_gcnn_classification_gat_training.py

    Requires precomputed data in the folder specified by DATA_ROOT.
    Outputs (checkpoints, logs) are saved to the folder specified by output_dir.
    See the Configuration section below for these and other settings.
"""

import os
import sys
from itertools import chain
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyhealth.datasets import EEGGCNNDataset
from pyhealth.datasets.collate import collate_temporal
from pyhealth.tasks import EEGGCNNClassification
from tqdm.autonotebook import trange

from pyhealth.trainer import Trainer, is_best

from pyhealth.models import EEGGATConvNet

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "precomputed_data")

EXPERIMENT_NAME = "psd_gat_shallow_ph"
BATCH_SIZE      = 512
NUM_EPOCHS      = 100
NUM_FOLDS       = 10 # set to 2 for a minimum train/val split; set to 10 for 10-fold CV
NUM_WORKERS     = 0       # macOS multiprocessing workaround; set >0 on Linux
SEED            = 42
LEARNING_RATE   = 1e-3   # used for both SGD and Adam
WEIGHT_DECAY    = 0.0

# Optimizer and LR scheduler selection.
# OPTIMIZER: "adam" or "sgd"
# LR_SCHEDULER: "plateau" — ReduceLROnPlateau, steps on val roc_auc (recommended)
#               "multistep" — MultiStepLR, decays at fixed epoch milestones
OPTIMIZER        = "adam"
LR_SCHEDULER     = "plateau"
PLATEAU_PATIENCE = 5     # epochs without improvement before LR is reduced
PLATEAU_FACTOR   = 0.5   # multiplicative LR reduction factor
TEST_RATIO      = 0.30
DROPOUT_VALUES       = [0.2]        # GAT dropout; e.g. [0.0, 0.2, 0.5]
ATTN_DROPOUT_VALUES  = [0.0]        # GAT attention dropout; e.g. [0.0, 0.3, 0.6]
MAX_PATIENTS: Optional[int] = None  # None uses the full dataset.
                                    # Set to an int (e.g. 20) to cap patients
                                    # for faster runs.

# Metrics reported after every validation epoch and in the fold summary.
METRICS = ["roc_auc", "pr_auc", "balanced_accuracy", "f1"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MapStyleDataset(torch.utils.data.Dataset):
    """Wraps a list of already-processed sample dicts for map-style access.

    SampleDataset is an IterableDataset (litdata.StreamingDataset), which
    prevents PyTorch DataLoader from accepting a custom sampler.  Materialising
    a fold's samples into a list restores map-style semantics so that
    WeightedRandomSampler works normally.

    Args:
        samples: List of processed sample dicts from a SampleDataset.
    """

    def __init__(self, samples: List[dict]) -> None:
        self._data = samples

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        return self._data[idx]


def patient_subset_samples(
    all_samples: List[dict],
    patient_ids: np.ndarray,
    patient_to_index: Dict[str, List[int]],
) -> List[dict]:
    """Return all processed samples for the given patient IDs.

    Uses direct list indexing on the pre-materialised sample list to avoid
    litdata StreamingDataset.subset(), which is designed for contiguous chunk
    ranges and does not reliably handle the non-contiguous index lists that
    arise from per-patient fold splits.

    Args:
        all_samples: Full list of materialised sample dicts.
        patient_ids: Array of patient ID strings to include.
        patient_to_index: Mapping from patient ID to list of sample indices.

    Returns:
        List of sample dicts belonging to the requested patients.
    """
    indices = list(chain.from_iterable(
        patient_to_index[pid]
        for pid in patient_ids
        if pid in patient_to_index
    ))
    return [all_samples[i] for i in indices]


def make_weighted_sampler(samples: List[dict]) -> WeightedRandomSampler:
    """Return a WeightedRandomSampler that up-samples the minority class.

    Computes per-class inverse-frequency weights so that each class is
    sampled with equal probability regardless of class imbalance.

    Args:
        samples: List of processed sample dicts, each containing a
            ``"label"`` tensor.

    Returns:
        A WeightedRandomSampler configured for balanced class sampling
        with replacement.
    """
    labels = np.array([int(s["label"].item()) for s in samples])
    classes, counts = np.unique(labels, return_counts=True)
    class_weight = {cls: 1.0 / cnt for cls, cnt in zip(classes, counts)}
    sample_weights = torch.tensor(
        [class_weight[lbl] for lbl in labels], dtype=torch.float32
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Trainer with MultiStepLR scheduler
# ---------------------------------------------------------------------------

class ScheduledTrainer(Trainer):
    """Trainer subclass that adds MultiStepLR and ReduceLROnPlateau support."""

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        epochs: int = 5,
        optimizer_class=torch.optim.Adam,
        optimizer_params: Optional[Dict] = None,
        scheduler_milestones: Optional[List[int]] = None,
        scheduler_gamma: float = 0.5,
        plateau_patience: int = 5,
        plateau_factor: float = 0.5,
        use_plateau_scheduler: bool = False,
        steps_per_epoch: Optional[int] = None,
        weight_decay: float = 0.0,
        max_grad_norm: Optional[float] = None,
        monitor: Optional[str] = None,
        monitor_criterion: str = "max",
        load_best_model_at_last: bool = True,
        patience: Optional[int] = None,
    ) -> None:
        """Run the training loop with an optional LR scheduler.

        Args:
            train_dataloader: DataLoader for the training split.
            val_dataloader: DataLoader for the validation split.
            test_dataloader: DataLoader for the test split.
            epochs: Number of training epochs. Defaults to 5.
            optimizer_class: Optimizer class to instantiate.
            optimizer_params: Keyword arguments forwarded to the optimizer.
            scheduler_milestones: Epoch indices for MultiStepLR decay.
                Ignored when ``use_plateau_scheduler=True``.
            scheduler_gamma: LR decay factor for MultiStepLR.
            plateau_patience: Epochs without improvement before
                ReduceLROnPlateau reduces the LR.
            plateau_factor: Multiplicative LR reduction for
                ReduceLROnPlateau.
            use_plateau_scheduler: If True, use ReduceLROnPlateau (steps on
                the monitored validation metric) instead of MultiStepLR.
            steps_per_epoch: Optimisation steps per epoch.
            weight_decay: L2 regularisation coefficient.
            max_grad_norm: Gradient clipping threshold.
            monitor: Metric name for best-model checkpointing / early stopping.
            monitor_criterion: ``"max"`` or ``"min"``.
            load_best_model_at_last: Reload best checkpoint after training.
            patience: Early-stopping patience in epochs.
        """
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3}

        param = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in param if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        scheduler = None
        if use_plateau_scheduler:
            plateau_mode = "max" if monitor_criterion == "max" else "min"
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=plateau_mode,
                patience=plateau_patience,
                factor=plateau_factor,
            )
            print(
                f"[Trainer] ReduceLROnPlateau: mode={plateau_mode}, "
                f"patience={plateau_patience}, factor={plateau_factor}"
            )
        elif scheduler_milestones is not None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=scheduler_milestones,
                gamma=scheduler_gamma,
            )
            print(
                f"[Trainer] MultiStepLR: milestones={scheduler_milestones}, "
                f"gamma={scheduler_gamma}"
            )

        data_iterator = iter(train_dataloader)
        best_score = (
            -1 * float("inf") if monitor_criterion == "max" else float("inf")
        )
        if steps_per_epoch is None:
            steps_per_epoch = len(train_dataloader)
        patience_counter = 0

        for epoch in range(epochs):
            training_loss = []
            self.model.zero_grad()
            self.model.train()
            for _ in trange(
                steps_per_epoch,
                desc=f"Epoch {epoch} / {epochs}",
                smoothing=0.05,
            ):
                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_dataloader)
                    data = next(data_iterator)
                output = self.model(**data)
                loss = output["loss"]
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad()
                training_loss.append(loss.item())

            if self.exp_path is not None:
                self.save_ckpt(os.path.join(self.exp_path, "last.ckpt"))

            avg_loss = sum(training_loss) / len(training_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"[Trainer] Epoch {epoch} loss={avg_loss:.4f}  lr={current_lr:.2e}")

            val_score = None
            if val_dataloader is not None:
                scores = self.evaluate(val_dataloader)
                print(f"[Trainer] Epoch {epoch} val scores: {scores}")
                if monitor is not None:
                    val_score = scores[monitor]
                    if is_best(best_score, val_score, monitor_criterion):
                        best_score = val_score
                        patience_counter = 0
                        if self.exp_path is not None:
                            self.save_ckpt(os.path.join(self.exp_path, "best.ckpt"))
                    else:
                        patience_counter += 1
                        if patience is not None and patience_counter >= patience:
                            print(f"[Trainer] Early stopping at epoch {epoch}")
                            break

            # Step scheduler after validation so plateau has the new score.
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_score is not None:
                        scheduler.step(val_score)
                else:
                    scheduler.step()

        if load_best_model_at_last and self.exp_path is not None and os.path.isfile(
            os.path.join(self.exp_path, "best.ckpt")
        ):
            self.load_ckpt(os.path.join(self.exp_path, "best.ckpt"))

        if test_dataloader is not None:
            scores = self.evaluate(test_dataloader)
            print(f"[Trainer] Test scores: {scores}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ------------------------------------------------------------------
    # Load dataset and build SampleDataset once (cached to disk)
    # ------------------------------------------------------------------
    print(f"[MAIN] Loading EEGGCNNDataset from: {DATA_ROOT}")
    dataset = EEGGCNNDataset(root=DATA_ROOT)
    dataset.stats()

    print("[MAIN] Building sample dataset (cached after first run)...")
    sample_ds = dataset.set_task(EEGGCNNClassification())

    # Materialise all processed samples into memory once.
    # This avoids litdata StreamingDataset.subset(), which is designed for
    # contiguous chunk ranges and is unreliable with the non-contiguous index
    # lists that arise from per-patient fold splits.
    print("[MAIN] Materialising sample dataset into memory...")
    all_samples = list(sample_ds)
    patient_to_index = sample_ds.patient_to_index

    # ------------------------------------------------------------------
    # Subject-level 70 / 30 train+val / held-out test split
    # ------------------------------------------------------------------
    all_patients = np.array(sorted(sample_ds.patient_to_index.keys()))
    if MAX_PATIENTS is not None:
        rng = np.random.default_rng(SEED)
        all_patients = rng.choice(
            all_patients,
            size=min(MAX_PATIENTS, len(all_patients)),
            replace=False,
        )
        all_patients = np.sort(all_patients)
        print(
            f"[MAIN] Capped to {len(all_patients)} patients "
            f"(MAX_PATIENTS={MAX_PATIENTS})"
        )

    train_val_patients, test_patients = train_test_split(
        all_patients, test_size=TEST_RATIO, random_state=SEED
    )

    print(
        f"\n[MAIN] {len(train_val_patients)} train+val patients | "
        f"{len(test_patients)} held-out test patients"
    )

    # ------------------------------------------------------------------
    # 10-fold cross-validation over train+val patients
    # ------------------------------------------------------------------
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_data")

    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    # (dropout, attn_dropout) -> list of per-fold val score dicts
    ablation_results: Dict[tuple, List[dict]] = {
        (d, a): [] for d in DROPOUT_VALUES for a in ATTN_DROPOUT_VALUES
    }

    for dropout in DROPOUT_VALUES:
        for attn_dropout in ATTN_DROPOUT_VALUES:
            print(
                f"\n[MAIN] ========== Dropout {dropout} | "
                f"Attn Dropout {attn_dropout} =========="
            )

            for fold_idx, (train_idx, val_idx) in enumerate(
                kfold.split(train_val_patients)
            ):
                print(f"\n[MAIN]   ===== Fold {fold_idx + 1}/{NUM_FOLDS} =====")

                train_patients = train_val_patients[train_idx]
                val_patients = train_val_patients[val_idx]

                train_samples = patient_subset_samples(
                    all_samples, train_patients, patient_to_index
                )
                val_samples = patient_subset_samples(
                    all_samples, val_patients, patient_to_index
                )

                print(
                    f"[MAIN]   Train windows: {len(train_samples)} | "
                    f"Val windows: {len(val_samples)}"
                )

                train_ds = _MapStyleDataset(train_samples)
                val_ds = _MapStyleDataset(val_samples)

                # Class-balanced training loader; sequential validation loader.
                train_loader = DataLoader(
                    train_ds,
                    batch_size=BATCH_SIZE,
                    sampler=make_weighted_sampler(train_samples),
                    num_workers=NUM_WORKERS,
                    pin_memory=False,
                    collate_fn=collate_temporal,
                )
                val_loader = DataLoader(
                    val_ds,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    pin_memory=False,
                    collate_fn=collate_temporal,
                )

                exp_name = (
                    f"{EXPERIMENT_NAME}"
                    f"_drop{int(dropout * 10)}"
                    f"_attn{int(attn_dropout * 10)}"
                    f"_fold_{fold_idx}"
                )

                # Fresh model for each fold, initialised from the full sample_ds schema.
                model = EEGGATConvNet(
                    dataset=sample_ds,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )

                trainer = ScheduledTrainer(
                    model=model,
                    metrics=METRICS,
                    device=None,          # auto: GPU if available, else CPU
                    enable_logging=True,
                    output_path=output_dir,
                    exp_name=exp_name,
                )

                optimizer_class = (
                    torch.optim.Adam if OPTIMIZER == "adam" else torch.optim.SGD
                )
                trainer.train(
                    train_dataloader=train_loader,
                    val_dataloader=val_loader,
                    epochs=NUM_EPOCHS,
                    optimizer_class=optimizer_class,
                    optimizer_params={"lr": LEARNING_RATE},
                    use_plateau_scheduler=(LR_SCHEDULER == "plateau"),
                    plateau_patience=PLATEAU_PATIENCE,
                    plateau_factor=PLATEAU_FACTOR,
                    scheduler_milestones=[i * 10 for i in range(1, 26)],
                    scheduler_gamma=0.5,
                    weight_decay=WEIGHT_DECAY,
                    max_grad_norm=None,
                    monitor="roc_auc",
                    monitor_criterion="max",
                    load_best_model_at_last=True,
                )

                val_scores = trainer.evaluate(val_loader)
                print(
                    f"[MAIN] Fold {fold_idx} dropout={dropout} "
                    f"attn_dropout={attn_dropout} val scores: {val_scores}"
                )
                ablation_results[(dropout, attn_dropout)].append(val_scores)

                ckpt_path = os.path.join(output_dir, f"{exp_name}.ckpt")
                trainer.save_ckpt(ckpt_path)
                print(f"[MAIN] Checkpoint saved: {ckpt_path}")

    # ------------------------------------------------------------------
    # Ablation summary
    # ------------------------------------------------------------------
    print(f"\n[MAIN] ========== Dropout Ablation Summary ==========")
    for (dropout, attn_dropout), fold_results in ablation_results.items():
        print(f"\n  Dropout={dropout} | Attn Dropout={attn_dropout}")
        for metric in fold_results[0].keys():
            vals = [r[metric] for r in fold_results if metric in r]
            print(f"    {metric:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
