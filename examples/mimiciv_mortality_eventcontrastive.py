from typing import Dict, List, Sequence, Tuple

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import EBCLModel
from pyhealth.tasks import MortalityPredictionMIMIC4


"""
Ablation study for EBCLModel using a real PyHealth MIMIC-IV mortality task.

Pipeline:
1. Load a MIMIC-IV dataset through PyHealth.
2. Apply the built-in MortalityPredictionMIMIC4 task.
3. Preprocess the task samples into EBCL-style inputs:
   - pre
   - post
   - pre_mask
   - post_mask
   - label
4. Perform a label-aware split so both train and validation sets contain
   both binary classes.
5. Wrap the processed samples with create_sample_dataset so EBCLModel can be
   trained without changing its input interface.
6. Compare several hyperparameter settings.

Notes:
- This script uses a real PyHealth dataset/task pipeline, but the EBCL inputs
  are derived through lightweight preprocessing because the built-in task emits
  conditions/procedures/drugs, while EBCLModel expects pre/post triplet inputs.
- This is intended as a rubric-compliant example/ablation script, not a
  faithful reproduction of the original EBCL cohort construction from the paper.

How to run:
    python3 examples/mimiciv_mortality_eventcontrastive.py

Before running:
- Update MIMIC4_PATH to your local MIMIC-IV path.
- Make sure your PyHealth version includes the MIMIC-IV dataset classes.
"""


MIMIC4_PATH = "/Users/davidwang/Desktop/mimic-iv-clinical-database-demo-2.2"
MAX_LEN = 32
MAX_FEATURES = 2048


def load_mimic4_base_dataset(mimic4_path: str):
    """Loads a MIMIC-IV PyHealth base dataset."""
    from pyhealth.datasets import MIMIC4EHRDataset

    return MIMIC4EHRDataset(
        root=mimic4_path,
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        dev=True,
    )


def flatten_codes(x) -> List[str]:
    """Flattens nested code structures into a simple list of strings."""
    if x is None:
        return []

    if isinstance(x, str):
        return [x]

    if not isinstance(x, Sequence):
        return [str(x)]

    flattened: List[str] = []
    for item in x:
        if isinstance(item, str):
            flattened.append(item)
        elif isinstance(item, Sequence):
            flattened.extend(flatten_codes(item))
        else:
            flattened.append(str(item))
    return flattened


def code_to_feature_id(
    code: str,
    vocab: Dict[str, int],
    max_features: int,
) -> int:
    """Maps a medical code string to an integer feature id."""
    if code not in vocab:
        next_id = len(vocab) + 1
        vocab[code] = min(next_id, max_features - 1)
    return vocab[code]


def build_triplets(
    codes: List[str],
    vocab: Dict[str, int],
    max_len: int,
    time_value: float,
    token_value: float,
    max_features: int,
) -> Tuple[List[List[float]], List[int]]:
    """Converts code strings into padded EBCL triplets and a mask."""
    triplets: List[List[float]] = []
    mask: List[int] = []

    for code in codes[:max_len]:
        feature_id = code_to_feature_id(code, vocab, max_features)
        triplets.append([time_value, float(feature_id), token_value])
        mask.append(1)

    while len(triplets) < max_len:
        triplets.append([0.0, 0.0, 0.0])
        mask.append(0)

    return triplets, mask


def preprocess_task_samples(
    task_dataset,
    max_len: int = MAX_LEN,
    max_features: int = MAX_FEATURES,
) -> List[Dict]:
    """Converts MortalityPredictionMIMIC4 samples into EBCL-style samples."""
    vocab: Dict[str, int] = {}
    processed_samples: List[Dict] = []

    for i in range(len(task_dataset)):
        sample = task_dataset[i]

        conditions = flatten_codes(sample.get("conditions", []))
        procedures = flatten_codes(sample.get("procedures", []))
        drugs = flatten_codes(sample.get("drugs", []))
        label = int(sample.get("mortality", 0))

        pre_codes = conditions + procedures
        post_codes = drugs

        pre, pre_mask = build_triplets(
            codes=pre_codes,
            vocab=vocab,
            max_len=max_len,
            time_value=0.1,
            token_value=1.0,
            max_features=max_features,
        )
        post, post_mask = build_triplets(
            codes=post_codes,
            vocab=vocab,
            max_len=max_len,
            time_value=0.2,
            token_value=1.0,
            max_features=max_features,
        )

        processed_samples.append(
            {
                "patient_id": sample.get("patient_id", f"patient-{i}"),
                "visit_id": sample.get(
                    "visit_id",
                    sample.get("hadm_id", f"visit-{i}"),
                ),
                "pre": pre,
                "post": post,
                "pre_mask": pre_mask,
                "post_mask": post_mask,
                "label": label,
            }
        )

    return processed_samples


def print_label_distribution(samples: List[Dict], name: str) -> None:
    """Prints label counts for a sample list."""
    count_0 = sum(1 for s in samples if int(s["label"]) == 0)
    count_1 = sum(1 for s in samples if int(s["label"]) == 1)
    print(f"{name} label distribution -> 0: {count_0}, 1: {count_1}")


def stratified_split_samples(
    samples: List[Dict],
    train_ratio: float = 0.8,
) -> Tuple[List[Dict], List[Dict]]:
    """Splits samples so train and val both contain both labels when possible."""
    class_0 = [s for s in samples if int(s["label"]) == 0]
    class_1 = [s for s in samples if int(s["label"]) == 1]

    if len(class_0) == 0 or len(class_1) == 0:
        raise ValueError(
            "The processed dataset contains only one label class overall."
        )

    g = torch.Generator().manual_seed(42)
    idx_0 = torch.randperm(len(class_0), generator=g).tolist()
    idx_1 = torch.randperm(len(class_1), generator=g).tolist()

    class_0 = [class_0[i] for i in idx_0]
    class_1 = [class_1[i] for i in idx_1]

    train_0 = max(1, int(len(class_0) * train_ratio))
    train_1 = max(1, int(len(class_1) * train_ratio))

    if len(class_0) - train_0 == 0 and len(class_0) > 1:
        train_0 -= 1
    if len(class_1) - train_1 == 0 and len(class_1) > 1:
        train_1 -= 1

    train_samples = class_0[:train_0] + class_1[:train_1]
    val_samples = class_0[train_0:] + class_1[train_1:]

    if len(val_samples) == 0:
        raise ValueError("Validation split is empty after stratified splitting.")

    train_perm = torch.randperm(len(train_samples), generator=g).tolist()
    val_perm = torch.randperm(len(val_samples), generator=g).tolist()

    train_samples = [train_samples[i] for i in train_perm]
    val_samples = [val_samples[i] for i in val_perm]

    return train_samples, val_samples


def build_ebcl_sample_dataset(samples: List[Dict], dataset_name: str):
    """Builds a PyHealth sample dataset in the EBCL input schema."""
    return create_sample_dataset(
        samples=samples,
        input_schema={
            "pre": "tensor",
            "post": "tensor",
            "pre_mask": "tensor",
            "post_mask": "tensor",
        },
        output_schema={"label": "binary"},
        dataset_name=dataset_name,
    )


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    """Moves tensor fields to device and normalizes mask types."""
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value

    if "pre_mask" in moved:
        moved["pre_mask"] = moved["pre_mask"].bool()
    if "post_mask" in moved:
        moved["post_mask"] = moved["post_mask"].bool()

    return moved


def train_one_epoch(model, loader, optimizer, device: torch.device) -> float:
    """Runs one training epoch."""
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()
        output = model(**batch)
        loss = output["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def evaluate_accuracy(model, loader, device: torch.device) -> float:
    """Evaluates binary accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            output = model(**batch)

            preds = (output["y_prob"] >= 0.5).long().view(-1)
            labels = output["y_true"].long().view(-1)

            correct += (preds == labels).sum().item()
            total += labels.numel()

    return correct / max(total, 1)


def run_config(
    dataset,
    train_loader,
    val_loader,
    config: Dict,
    device: torch.device,
) -> Dict:
    """Trains and evaluates one hyperparameter configuration."""
    model = EBCLModel(
        dataset=dataset,
        num_features=MAX_FEATURES,
        d_model=config["d_model"],
        n_heads=4,
        n_layers=1,
        projection_dim=config["projection_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        mode="finetune",
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    train_losses = []
    for _ in range(config["epochs"]):
        epoch_loss = train_one_epoch(model, train_loader, optimizer, device)
        train_losses.append(epoch_loss)

    val_acc = evaluate_accuracy(model, val_loader, device)

    return {
        "name": config["name"],
        "lr": config["lr"],
        "d_model": config["d_model"],
        "hidden_dim": config["hidden_dim"],
        "projection_dim": config["projection_dim"],
        "dropout": config["dropout"],
        "final_train_loss": train_losses[-1],
        "val_accuracy": val_acc,
    }


def print_results_table(results: List[Dict]) -> None:
    """Prints a compact comparison table."""
    print("\nAblation Results: EBCLModel on MIMIC-IV Mortality Task")
    print("-" * 90)
    print(
        f"{'Config':<18}"
        f"{'LR':<10}"
        f"{'d_model':<10}"
        f"{'hidden':<10}"
        f"{'proj':<8}"
        f"{'dropout':<10}"
        f"{'train_loss':<14}"
        f"{'val_acc':<10}"
    )
    print("-" * 90)

    for row in results:
        print(
            f"{row['name']:<18}"
            f"{row['lr']:<10.4g}"
            f"{row['d_model']:<10}"
            f"{row['hidden_dim']:<10}"
            f"{row['projection_dim']:<8}"
            f"{row['dropout']:<10.2f}"
            f"{row['final_train_loss']:<14.4f}"
            f"{row['val_accuracy']:<10.4f}"
        )
    print("-" * 90)


def main() -> None:
    """Runs the MIMIC-IV mortality ablation study."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading MIMIC-IV base dataset...")
    base_dataset = load_mimic4_base_dataset(MIMIC4_PATH)

    print("Applying MortalityPredictionMIMIC4 task...")
    task_dataset = base_dataset.set_task(MortalityPredictionMIMIC4())

    print("Preprocessing task samples into EBCL input format...")
    all_processed_samples = preprocess_task_samples(task_dataset)

    print_label_distribution(all_processed_samples, "All samples")

    print("Performing stratified split...")
    train_samples, val_samples = stratified_split_samples(
        all_processed_samples,
        train_ratio=0.8,
    )

    print_label_distribution(train_samples, "Train")
    print_label_distribution(val_samples, "Validation")

    train_dataset = build_ebcl_sample_dataset(
        train_samples,
        dataset_name="mimic4_mortality_eventcontrastive_train",
    )
    val_dataset = build_ebcl_sample_dataset(
        val_samples,
        dataset_name="mimic4_mortality_eventcontrastive_val",
    )

    train_loader = get_dataloader(train_dataset, batch_size=16, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=16, shuffle=False)

    configs = [
        {
            "name": "baseline",
            "lr": 1e-3,
            "d_model": 32,
            "hidden_dim": 32,
            "projection_dim": 16,
            "dropout": 0.1,
            "epochs": 5,
        },
        {
            "name": "larger_hidden",
            "lr": 1e-3,
            "d_model": 64,
            "hidden_dim": 64,
            "projection_dim": 32,
            "dropout": 0.1,
            "epochs": 5,
        },
        {
            "name": "higher_dropout",
            "lr": 1e-3,
            "d_model": 32,
            "hidden_dim": 32,
            "projection_dim": 16,
            "dropout": 0.3,
            "epochs": 5,
        },
        {
            "name": "lower_lr",
            "lr": 3e-4,
            "d_model": 32,
            "hidden_dim": 32,
            "projection_dim": 16,
            "dropout": 0.1,
            "epochs": 5,
        },
    ]

    results = []
    for config in configs:
        print(f"Running config: {config['name']}")
        result = run_config(
            dataset=train_dataset,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
        )
        results.append(result)

    results.sort(key=lambda x: x["val_accuracy"], reverse=True)
    print_results_table(results)

if __name__ == "__main__":
    main()