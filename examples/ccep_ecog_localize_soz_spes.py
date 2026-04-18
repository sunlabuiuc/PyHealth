# =============================================================================
# Experimental Setup
#
# Commands run (5-fold patient-level cross-validation, seed=42):
#
#   python examples/ceep_ecog_localize_soz_spes.py --model cnn-divergent
#   python examples/ceep_ecog_localize_soz_spes.py --model cnn-convergent
#   python examples/ccep_ecog_localize_soz_spes.py --model cnn-transformer
#   python examples/ccep_ecog_localize_soz_spes.py --model cnn-transformer --lr 1e-4 --dropout 0.5
#   python examples/ccep_ecog_localize_soz_spes.py --model cnn-transformer-ablation
#
# cnn-transformer is run twice: once with paper-tuned hyperparameters, and once with hyperparameters
# matched to the ablation (lr=1e-4, dropout=0.5) to enable a fair comparison.
# cnn-transformer-ablation uses the same matched hyperparameters but removes the std response mode
# and the MLP prefix from the convergent encoder, isolating their contribution.
# =============================================================================
import argparse
from functools import partial

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve
import torch
from torch.utils.data import DataLoader

from pyhealth.datasets import CCEPECoGDataset
from pyhealth.models import SPESResNet, SPESTransformer
from pyhealth.tasks.localize_soz import LocalizeSOZ
from pyhealth.trainer import Trainer


ROOT = "./data/ds004080"
CACHE_DIR = "./ccep_ecog"


# NOTE: MODEL_PRESETS hyperparameters for cnn-divergent, cnn-convergent, and cnn-transformer
# were optimized via Optuna in the original paper (https://proceedings.mlr.press/v252/norris24a.html)
# and pulled directly from the paper's codebase (https://github.com/norrisjamie23/Localising_SOZ_from_SPES).
MODEL_PRESETS = {
    "cnn-divergent": {
        "learning_rate": 0.003962831229235175,
        "model_kwargs": {
            "dropout_rate": 0.21763415739071962,
            "input_channels": 49,
        },
    },
    "cnn-convergent": {
        "learning_rate": 0.001289042623854371,
        "model_kwargs": {
            "dropout_rate": 0.44374819954858546,
            "input_channels": 37,
        },
    },
    "cnn-transformer": {
        "learning_rate": 0.003368045116199473,
        "model_kwargs": {
            "dropout_rate": 0.4391902174353594,
            "embedding_dim": 2**4,
            "num_layers": 2**1,
        },
    },
    # Ablation: removes the std response mode and MLP prefix to isolate
    # the contribution of trial variability and the hybrid embedding.
    "cnn-transformer-ablation": {
        "learning_rate": 1e-4,
        "model_kwargs": {
            "dropout_rate": 0.5,
            "embedding_dim": 2**4,
            "num_layers": 2**1,
        },
    },
}


def pad_tensor_to_shape(value, shape):
    pad = []
    for current, target in zip(reversed(value.shape), reversed(shape)):
        pad.extend([0, target - current])
    return torch.nn.functional.pad(value, pad)


def compute_norm_stats(dataset, keys=("X_stim", "X_recording")):
    """Compute normalization statistics from the training dataset only.

    This must be called exclusively on the training split and the resulting
    statistics applied to all splits (train, val, test). Computing stats on
    the full dataset would leak val/test distribution information into
    normalization, invalidating the evaluation.

    Distances are stored at position 0 of the last dim in cached tensors.
    Stats are computed using the paper's per-sample averaging approach: mean
    and std are averaged across samples rather than computed globally, to avoid
    samples with more channels dominating the statistics.
    """
    dist_values = []
    ts_sample_means = []
    ts_sample_stds = []

    for sample in dataset:
        for key in keys:
            if key not in sample:
                continue
            x = sample[key]  # (modes, chans, T+1), distance at position 0
            dist = x[0, :, 0]
            valid = dist > 0
            if valid.any():
                dist_values.extend(dist[valid].tolist())
            ts = x[:, :, 1:]  # (modes, chans, T)
            ts_std = ts.std(dim=-1)  # (modes, chans)
            nonzero = ts_std > 0
            if nonzero.any():
                ts_sample_means.append(ts[nonzero.unsqueeze(-1).expand_as(ts)].mean().item())
                ts_sample_stds.append(ts_std[nonzero].mean().item())

    return {
        "mean_dist": float(np.mean(dist_values)) if dist_values else 0.0,
        "std_dist": float(np.std(dist_values)) if len(dist_values) > 1 else 1.0,
        "mean_ts": float(np.mean(ts_sample_means)) if ts_sample_means else 0.0,
        "std_ts": float(np.mean(ts_sample_stds)) if ts_sample_stds else 1.0,
    }


def normalize_spes_tensor(x, norm_stats):
    """Apply z-score normalization to a single SPES tensor using pre-computed stats.

    Distances (position 0 of last dim) and time series (positions 1:) are
    normalized separately, since they have different units and scales. Only
    non-zero-padded entries are normalized; padded channels (zero distance,
    zero-std time series) are left as zero so the model can distinguish real
    data from padding.
    """
    x = x.clone()
    dist_mask = x[..., 0] > 0
    if dist_mask.any():
        x[..., 0][dist_mask] = (
            x[..., 0][dist_mask] - norm_stats["mean_dist"]
        ) / norm_stats["std_dist"]
    ts = x[..., 1:]
    ts_std = ts.std(dim=-1)
    ts_mask = (ts_std > 0).unsqueeze(-1).expand_as(ts)
    if ts_mask.any():
        x[..., 1:][ts_mask] = (
            x[..., 1:][ts_mask] - norm_stats["mean_ts"]
        ) / norm_stats["std_ts"]
    return x


def collate_spes_batch(batch, norm_stats=None):
    """Collate a batch of SPES samples, padding variable-length tensors and applying normalization.

    NOTE: PyHealth's built-in get_dataloader (pyhealth.datasets.utils) is not
    used here for two reasons that are fundamental to this task:
    
    1. Variable-shape inputs: X_stim and X_recording have a variable number
       of trials (rows) per electrode, determined by how many stimulation
       events were recorded for each channel. This varies across electrodes
       and patients, so samples within a batch cannot be stacked without
       padding. PyHealth's default collate_fn_dict_with_padding does not
       handle this multi-dimensional, field-specific padding.
    
    2. Per-fold normalization at collate time: z-score statistics are
       computed from the training split only and must be injected into the
       collate function via functools.partial. PyHealth's get_dataloader
       accepts no such hook, so normalization would have to happen elsewhere
       (e.g., in the model or dataset), breaking the clean separation between
       preprocessing and model logic.
    """
    collated = {}
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        if key in {"X_stim", "X_recording"}:
            max_shape = tuple(max(value.shape[dim] for value in values) for dim in range(values[0].dim()))
            stacked = torch.stack(
                [pad_tensor_to_shape(value, max_shape) for value in values]
            )
            if norm_stats is not None:
                stacked = normalize_spes_tensor(stacked, norm_stats)
            collated[key] = stacked
        elif isinstance(values[0], torch.Tensor):
            if all(value.shape == values[0].shape for value in values):
                collated[key] = torch.stack(values)
            elif values[0].dim() == 0:
                collated[key] = torch.stack(values)
            else:
                max_shape = tuple(max(value.shape[dim] for value in values) for dim in range(values[0].dim()))
                collated[key] = torch.stack(
                    [pad_tensor_to_shape(value, max_shape) for value in values]
                )
        else:
            collated[key] = values
    return collated


def get_spes_dataloader(dataset, batch_size, shuffle=False, norm_stats=None):
    """Create a DataLoader with SPES-specific collation and normalization.

    norm_stats must be computed from the training split only (via
    compute_norm_stats) and passed to all splits so that val and test are
    normalized using training-set statistics, preventing data leakage.
    """
    dataset.set_shuffle(shuffle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_spes_batch, norm_stats=norm_stats),
    )


def build_model(model_name, sample_dataset, pos_weight=None, dropout_rate=None):
    model_kwargs = dict(MODEL_PRESETS[model_name]["model_kwargs"])
    if dropout_rate is not None:
        model_kwargs["dropout_rate"] = dropout_rate
    if model_name == "cnn-divergent":
        return SPESResNet(
            dataset=sample_dataset,
            input_type="divergent",
            pos_weight=pos_weight,
            **model_kwargs,
        )
    if model_name == "cnn-convergent":
        return SPESResNet(
            dataset=sample_dataset,
            input_type="convergent",
            pos_weight=pos_weight,
            **model_kwargs,
        )
    if model_name == "cnn-transformer":
        return SPESTransformer(
            dataset=sample_dataset,
            net_configs=[
                {"type": "convergent", "mean": True, "std": True},
            ],
            pos_weight=pos_weight,
            **model_kwargs,
        )
    if model_name == "cnn-transformer-ablation":
        return SPESTransformer(
            dataset=sample_dataset,
            net_configs=[
                {"type": "convergent", "mean": True, "std": False},
            ],
            mlp_embedding=False,
            pos_weight=pos_weight,
            **model_kwargs,
        )
    raise ValueError(f"Unknown model: {model_name}")


def compute_pos_weight(dataset):
    labels = np.array([int(sample["soz"].item()) for sample in dataset])
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0:
        return 1.0
    return negatives / positives


def split_by_patient_kfold(dataset, fold=0, n_splits=5, seed=0):
    patient_ids = np.array(sorted(dataset.patient_to_index.keys()))
    if len(patient_ids) < n_splits:
        raise ValueError(
            f"Need at least n_splits patients; got {len(patient_ids)} patients "
            f"and n_splits={n_splits}."
        )

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(kfold.split(patient_ids))
    fold = fold % n_splits

    test_patient_idx = splits[fold][1]
    val_patient_idx = splits[(fold + 1) % n_splits][1]
    train_patient_idx = np.array(
        sorted(set(splits[fold][0]) - set(val_patient_idx))
    )

    def patient_indices_to_sample_indices(patient_idx):
        sample_indices = []
        for index in patient_idx:
            sample_indices.extend(dataset.patient_to_index[patient_ids[index]])
        return sample_indices

    train_dataset = dataset.subset(patient_indices_to_sample_indices(train_patient_idx))
    val_dataset = dataset.subset(patient_indices_to_sample_indices(val_patient_idx))
    test_dataset = dataset.subset(patient_indices_to_sample_indices(test_patient_idx))
    return train_dataset, val_dataset, test_dataset


def youden_score(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)

    true_positive = int(((y_true == 1) & (y_pred == 1)).sum())
    true_negative = int(((y_true == 0) & (y_pred == 0)).sum())
    false_positive = int(((y_true == 0) & (y_pred == 1)).sum())
    false_negative = int(((y_true == 1) & (y_pred == 0)).sum())

    sensitivity = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else float("nan")
    )
    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive > 0
        else float("nan")
    )
    return sensitivity, specificity, sensitivity + specificity - 1


def select_youden_threshold(y_true, y_prob):
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return 0.5

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_prob)
    youden_values = true_positive_rate - false_positive_rate
    threshold = thresholds[int(np.argmax(youden_values))]
    if np.isfinite(threshold):
        return float(threshold)
    return 0.5


def safe_roc_auc(y_true, y_prob):
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_prob)


def compute_soz_metrics(y_true, y_prob, patient_ids, threshold=0.5):
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1)
    patient_ids = np.asarray(patient_ids)

    patient_aucs = []
    baselines = []
    youdens = []
    specificities = []
    sensitivities = []

    for patient_id in np.unique(patient_ids):
        patient_mask = patient_ids == patient_id
        patient_true = y_true[patient_mask]
        patient_prob = y_prob[patient_mask]
        patient_pred = patient_prob > threshold

        patient_aucs.append(safe_roc_auc(patient_true, patient_prob))
        baselines.append(np.mean(patient_true))

        sensitivity, specificity, youden = youden_score(patient_true, patient_pred)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        youdens.append(youden)

    return {
        "Baseline": np.nanmean(baselines),
        "AUROC": np.nanmean(patient_aucs),
        "Specificity": np.nanmean(specificities),
        "Sensitivity": np.nanmean(sensitivities),
        "Youden": np.nanmean(youdens),
        "Youden threshold": threshold,
    }


def run_fold(args, sample_dataset, fold):
    train_dataset, val_dataset, test_dataset = split_by_patient_kfold(
        sample_dataset,
        fold=fold,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    print("\nPatient-level k-fold split sizes:")
    print(f"  fold: {fold} / {args.n_splits}")
    print(f"  train: {len(train_dataset)}")
    print(f"  val: {len(val_dataset)}")
    print(f"  test: {len(test_dataset)}")

    if min(len(train_dataset), len(val_dataset), len(test_dataset)) == 0:
        print("\nSkipping fold because at least one split is empty.")
        return None

    pos_weight = compute_pos_weight(train_dataset)
    model = build_model(args.model, sample_dataset, pos_weight=pos_weight, dropout_rate=args.dropout)
    print(f"\nInitialized model: {model.__class__.__name__} ({args.model})")
    learning_rate = (
        args.lr
        if args.lr is not None
        else MODEL_PRESETS[args.model]["learning_rate"]
    )
    print(f"Using learning rate: {learning_rate}")
    print(f"Using positive class weight: {pos_weight}")

    print("\nComputing normalization statistics from training set...")
    norm_stats = compute_norm_stats(train_dataset)
    print(f"  mean_dist={norm_stats['mean_dist']:.4f}, std_dist={norm_stats['std_dist']:.4f}")
    print(f"  mean_ts={norm_stats['mean_ts']:.6f}, std_ts={norm_stats['std_ts']:.6f}")

    train_loader = get_spes_dataloader(
        train_dataset,
        batch_size=min(args.batch_size, len(train_dataset)),
        shuffle=True,
        norm_stats=norm_stats,
    )
    val_loader = get_spes_dataloader(
        val_dataset,
        batch_size=min(args.batch_size, len(val_dataset)),
        shuffle=False,
        norm_stats=norm_stats,
    )
    test_loader = get_spes_dataloader(
        test_dataset,
        batch_size=min(args.batch_size, len(test_dataset)),
        shuffle=False,
        norm_stats=norm_stats,
    )

    trainer = Trainer(
        model=model,
        device=args.device,
        metrics=["roc_auc"],
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        optimizer_class=torch.optim.AdamW,
        optimizer_params={"lr": learning_rate},
        monitor="roc_auc",
        monitor_criterion="max",
        patience=args.patience,
    )

    val_y_true, val_y_prob, _ = trainer.inference(val_loader)
    threshold = select_youden_threshold(val_y_true, val_y_prob)
    print(f"Using decision threshold: {threshold}")

    y_true, y_prob, test_loss, patient_ids = trainer.inference(
        test_loader,
        return_patient_ids=True,
    )
    results = compute_soz_metrics(
        y_true,
        y_prob,
        patient_ids,
        threshold=threshold,
    )
    results["loss"] = test_loss
    print("\nTest results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    results["model"] = args.model
    results["seed"] = args.seed
    results["fold"] = fold
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="cnn-transformer",
        choices=[
            "cnn-divergent",
            "cnn-convergent",
            "cnn-transformer",
            "cnn-transformer-ablation",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override the selected model preset learning rate.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Override the selected model preset dropout rate.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Run one fold only. Defaults to all folds.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    dataset = CCEPECoGDataset(
        root=ROOT,
        dev=False,
        cache_dir=CACHE_DIR,
        num_workers=1,
    )

    print("\nBuilding PyHealth LocalizeSOZ sample dataset...")
    sample_dataset = dataset.set_task(LocalizeSOZ(), num_workers=1)

    print(f"PyHealth task name: {sample_dataset.task_name}")
    print(f"Total electrode samples: {len(sample_dataset)}")

    positive_task_patients = set()
    positive_electrodes = 0
    for sample in sample_dataset:
        patient_id = sample["patient_id"]
        if int(sample["soz"].item()) == 1:
            positive_task_patients.add(patient_id)
            positive_electrodes += 1
            
    print(f"Patients with positive SOZ electrode samples: {len(positive_task_patients)}")
    print(f"Positive SOZ electrode samples: {positive_electrodes}")

    if len(sample_dataset):
        sample = sample_dataset[0]
        print("\nFirst processed PyHealth sample:")
        print(f"  patient_id: {sample['patient_id']}")
        print(f"  record_id: {sample['record_id']}")
        print(f"  channel: {sample['channel']}")
        print(f"  soz shape/value: {tuple(sample['soz'].shape)} / {sample['soz'].tolist()}")
        print(f"  electrode_lobes shape: {tuple(sample['electrode_lobes'].shape)}")
        print(f"  electrode_coords shape: {tuple(sample['electrode_coords'].shape)}")
        print(f"  X_stim shape: {tuple(sample['X_stim'].shape)}")
        print(f"  X_recording shape: {tuple(sample['X_recording'].shape)}")
        print("  X_stim mode axis: 0=mean, 1=std")
        print("  X_recording mode axis: 0=mean, 1=std")

    if not len(sample_dataset):
        return

    folds = [args.fold % args.n_splits] if args.fold is not None else range(args.n_splits)
    fold_results = []
    for fold in folds:
        print(f"\nSeed {args.seed}, fold {fold + 1}")
        result = run_fold(args, sample_dataset, fold)
        if result is not None:
            fold_results.append(result)

    if not fold_results:
        print("\nNo fold results to summarize.")
        return

    metric_names = [
        "Baseline",
        "AUROC",
        "Specificity",
        "Sensitivity",
        "Youden",
        "loss",
    ]
    if len(fold_results) > 1:
        print("\nMean and standard deviation across folds:")
        for metric in metric_names:
            values = np.array([result[metric] for result in fold_results], dtype=float)
            print(f"  {metric}: {np.nanmean(values):.4f} +/- {np.nanstd(values):.4f}")


if __name__ == "__main__":
    main()
