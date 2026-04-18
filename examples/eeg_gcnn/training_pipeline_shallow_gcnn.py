"""PyHealth 2.0 training pipeline for shallow EEG-GCNN.

10-fold cross-validation using PyHealth 2.0 Trainer, EEGGCNNDataset, and
the shallow EEGGraphConvNet (2 GCN layers).  Mirrors the subject-level 70/30
train+val / heldout-test split used in the deep pipeline.

Dataset loading:
  EEGGCNNDataset.set_task() is called once to build and cache a SampleDataset
  on disk.  Subject-level 70/30 and fold splits are then made by indexing into
  SampleDataset.patient_to_index — no data is re-loaded between folds.

Checkpoints are saved as pure state-dicts:
    {EXPERIMENT_NAME}_fold_{fold_idx}.ckpt

Load with:
    model.load_state_dict(torch.load("psd_gnn_shallow_ph_alpha0.50_fold_0.ckpt"))

Usage (from the examples/eeg_gcnn directory):
    conda activate pyhealth (assuming PyHealth is installed in this conda env)
    python training_pipeline_shallow_gcnn.py

    Requires precomputed data in the folder specified by DATA_ROOT.
    Outputs (checkpoints, logs) are saved to the folder specified by output_dir.
    See the Configuration section below for these and other settings.

Ablations:
    Edge weight mix (ALPHA):
        Controls the blend between geodesic distance and coherence-based
        edge weights. Set ALPHA in the Configuration section:
            ALPHA = 1.0   # geodesic only
            ALPHA = 0.0   # coherence only
            ALPHA = 0.5   # equal mix (default)
        Note: changing ALPHA requires clearing the PyHealth dataset cache,
        which this script does automatically on each run.

    Patient subset (MAX_PATIENTS):
        Set MAX_PATIENTS to an integer to limit the number of patients used,
        which is useful for quick smoke-tests:
            MAX_PATIENTS = 20   # fast dev run
            MAX_PATIENTS = None # full dataset (default)
"""

import os
import shutil
import sys
from itertools import chain
from pathlib import Path
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

from pyhealth.models import EEGGraphConvNet

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "precomputed_data")

ALPHA           = 0.5     # edge weight mix: 1.0=geodesic only, 0.0=coherence only
EXPERIMENT_NAME = f"psd_gnn_shallow_ph_alpha{ALPHA:.2f}"
BATCH_SIZE      = 512
NUM_EPOCHS      = 100
NUM_FOLDS       = 10   # minimum 2 (one train/val split); 10 for full 10-fold CV
NUM_WORKERS     = 0       # macOS multiprocessing workaround; set >0 on Linux
SEED            = 42
LEARNING_RATE   = 0.01
WEIGHT_DECAY    = 0.0
TEST_RATIO      = 0.30
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
    """Trainer subclass that adds MultiStepLR scheduler support.

    Identical to the base Trainer but calls scheduler.step() after each
    epoch when scheduler_milestones is provided.
    """

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        epochs: int = 5,
        optimizer_class=torch.optim.Adam,
        optimizer_params: Optional[Dict] = None,
        scheduler_milestones: Optional[List[int]] = None,
        scheduler_gamma: float = 0.1,
        steps_per_epoch: Optional[int] = None,
        weight_decay: float = 0.0,
        max_grad_norm: Optional[float] = None,
        monitor: Optional[str] = None,
        monitor_criterion: str = "max",
        load_best_model_at_last: bool = True,
        patience: Optional[int] = None,
    ) -> None:
        """Run the training loop with an optional MultiStepLR scheduler.

        Args:
            train_dataloader: DataLoader for the training split.
            val_dataloader: DataLoader for the validation split. If provided,
                validation scores are logged after each epoch.
            test_dataloader: DataLoader for the test split. If provided,
                scores are reported at the end of training.
            epochs: Number of training epochs. Defaults to 5.
            optimizer_class: Optimizer class to instantiate. Defaults to
                ``torch.optim.Adam``.
            optimizer_params: Keyword arguments forwarded to the optimizer
                (e.g. ``{"lr": 0.01}``). Defaults to ``{"lr": 1e-3}``.
            scheduler_milestones: Epoch indices at which to decay the LR.
                If None, no scheduler is used.
            scheduler_gamma: Multiplicative LR decay factor applied at each
                milestone. Defaults to 0.1.
            steps_per_epoch: Number of optimisation steps per epoch. Defaults
                to ``len(train_dataloader)``.
            weight_decay: L2 regularisation coefficient. Defaults to 0.0.
            max_grad_norm: Maximum gradient norm for clipping. If None,
                no clipping is applied.
            monitor: Metric name to track for best-model checkpointing and
                early stopping (e.g. ``"roc_auc"``).
            monitor_criterion: ``"max"`` or ``"min"`` depending on whether
                higher or lower values of ``monitor`` are better.
                Defaults to ``"max"``.
            load_best_model_at_last: If True, reload the best checkpoint
                after training completes. Defaults to True.
            patience: Number of epochs without improvement before early
                stopping. If None, no early stopping is applied.
        """
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3}

        param = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in param
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        scheduler = None
        if scheduler_milestones is not None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=scheduler_milestones,
                gamma=scheduler_gamma,
            )
            print(
                f"[Trainer] MultiStepLR scheduler: "
                f"milestones={scheduler_milestones}, gamma={scheduler_gamma}"
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

            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                print(f"[Trainer] Epoch {epoch}: LR={current_lr:.6f}")

            if self.exp_path is not None:
                self.save_ckpt(os.path.join(self.exp_path, "last.ckpt"))

            avg_loss = sum(training_loss) / len(training_loss)
            print(f"[Trainer] Epoch {epoch} loss={avg_loss:.4f}")

            if val_dataloader is not None:
                scores = self.evaluate(val_dataloader)
                print(f"[Trainer] Epoch {epoch} val scores: {scores}")
                if monitor is not None:
                    score = scores[monitor]
                    if is_best(best_score, score, monitor_criterion):
                        best_score = score
                        patience_counter = 0
                        if self.exp_path is not None:
                            self.save_ckpt(
                                os.path.join(self.exp_path, "best.ckpt")
                            )
                    else:
                        patience_counter += 1
                        if patience is not None and patience_counter >= patience:
                            print(f"[Trainer] Early stopping at epoch {epoch}")
                            break

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
    # Clear PyHealth dataset cache so EEGGCNNDataset rebuilds from the
    # correct alpha-specific npy files rather than returning stale data.
    # Both global_event_df.parquet (stores npy paths) and tasks/ (stores
    # processed samples) must be cleared when alpha changes.
    # ------------------------------------------------------------------
    for cache_root in [
        Path.home() / "Library" / "Caches" / "pyhealth",
        Path.home() / ".cache" / "pyhealth",
    ]:
        for dataset_dir in cache_root.glob("*"):
            if dataset_dir.is_dir():
                shutil.rmtree(dataset_dir, ignore_errors=True)
                print(f"[MAIN] Cleared PyHealth dataset cache: {dataset_dir}")

    # ------------------------------------------------------------------
    # Load dataset and build SampleDataset once (cached to disk)
    # ------------------------------------------------------------------
    print(f"[MAIN] Loading EEGGCNNDataset from: {DATA_ROOT}")
    dataset = EEGGCNNDataset(root=DATA_ROOT, alpha=ALPHA)
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

    if NUM_FOLDS < 2:
        raise ValueError(f"NUM_FOLDS must be at least 2, got {NUM_FOLDS}.")

    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    fold_results: List[Dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        kfold.split(train_val_patients)
    ):
        print(f"\n[MAIN] ========== Fold {fold_idx + 1}/{NUM_FOLDS} ==========")

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

        # Fresh model for each fold, initialised from the full sample_ds schema.
        model = EEGGraphConvNet(dataset=sample_ds)

        trainer = ScheduledTrainer(
            model=model,
            metrics=METRICS,
            device=None,          # auto: GPU if available, else CPU
            enable_logging=True,
            output_path=output_dir,
            exp_name=f"{EXPERIMENT_NAME}_fold_{fold_idx}",
        )

        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=NUM_EPOCHS,
            optimizer_class=torch.optim.SGD,
            optimizer_params={"lr": LEARNING_RATE},
            scheduler_milestones=[i * 10 for i in range(1, 26)],
            scheduler_gamma=0.1,
            weight_decay=WEIGHT_DECAY,
            max_grad_norm=None,
            monitor="roc_auc",
            monitor_criterion="max",
            load_best_model_at_last=True,
        )

        val_scores = trainer.evaluate(val_loader)
        print(f"[MAIN] Fold {fold_idx} final val scores: {val_scores}")
        fold_results.append(val_scores)

        ckpt_path = os.path.join(output_dir, f"{EXPERIMENT_NAME}_fold_{fold_idx}.ckpt")
        trainer.save_ckpt(ckpt_path)
        print(f"[MAIN] Checkpoint saved: {ckpt_path}")

    # ------------------------------------------------------------------
    # Cross-fold summary
    # ------------------------------------------------------------------
    print(f"\n[MAIN] ========== {NUM_FOLDS}-Fold CV Summary ==========")
    for metric in fold_results[0].keys():
        vals = [r[metric] for r in fold_results if metric in r]
        print(f"  {metric:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
