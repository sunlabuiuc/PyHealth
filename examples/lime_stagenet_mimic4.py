# %% Loading MIMIC-IV dataset with LIME explanations
from pathlib import Path

import polars as pl
import torch

from pyhealth.datasets import (
    MIMIC4EHRDataset,
    get_dataloader,
    load_processors,
    split_by_patient,
)
from pyhealth.interpret.methods import LimeExplainer
from pyhealth.models import StageNet
from pyhealth.tasks import MortalityPredictionStageNetMIMIC4


def load_icd_description_map(dataset_root: str) -> dict:
    """Load ICD code → long title mappings from MIMIC-IV reference tables."""
    mapping = {}
    root_path = Path(dataset_root).expanduser()
    diag_path = root_path / "hosp" / "d_icd_diagnoses.csv.gz"
    proc_path = root_path / "hosp" / "d_icd_procedures.csv.gz"

    icd_dtype = {"icd_code": pl.Utf8, "long_title": pl.Utf8}

    if diag_path.exists():
        diag_df = pl.read_csv(
            diag_path,
            columns=["icd_code", "long_title"],
            dtypes=icd_dtype,
        )
        mapping.update(
            zip(diag_df["icd_code"].to_list(), diag_df["long_title"].to_list())
        )

    if proc_path.exists():
        proc_df = pl.read_csv(
            proc_path,
            columns=["icd_code", "long_title"],
            dtypes=icd_dtype,
        )
        mapping.update(
            zip(proc_df["icd_code"].to_list(), proc_df["long_title"].to_list())
        )

    return mapping


LAB_CATEGORY_NAMES = MortalityPredictionStageNetMIMIC4.LAB_CATEGORY_NAMES


def move_batch_to_device(batch, target_device):
    """Move all tensors in batch to target device."""
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(target_device)
        elif isinstance(value, tuple):
            moved[key] = tuple(v.to(target_device) for v in value)
        else:
            moved[key] = value
    return moved


def decode_token(idx: int, processor, feature_key: str, icd_code_to_desc: dict):
    """Decode token index to human-readable string."""
    if processor is None or not hasattr(processor, "code_vocab"):
        return str(idx)
    reverse_vocab = {index: token for token, index in processor.code_vocab.items()}
    token = reverse_vocab.get(idx, f"<UNK:{idx}>")

    if feature_key == "icd_codes" and token not in {"<unk>", "<pad>"}:
        desc = icd_code_to_desc.get(token)
        if desc:
            return f"{token}: {desc}"

    return token


def unravel(flat_index: int, shape: torch.Size):
    """Convert flat index to multi-dimensional coordinates."""
    coords = []
    remaining = flat_index
    for dim in reversed(shape):
        coords.append(remaining % dim)
        remaining //= dim
    return list(reversed(coords))


def print_top_attributions(
    attributions,
    batch,
    processors,
    top_k: int = 10,
    icd_code_to_desc: dict = None,
    method_name: str = "LIME",
):
    """Print top-k most important features from LIME attributions."""
    if icd_code_to_desc is None:
        icd_code_to_desc = {}

    for feature_key, attr in attributions.items():
        attr_cpu = attr.detach().cpu()
        if attr_cpu.dim() == 0 or attr_cpu.size(0) == 0:
            continue

        feature_input = batch[feature_key]
        if isinstance(feature_input, tuple):
            feature_input = feature_input[1]
        feature_input = feature_input.detach().cpu()

        flattened = attr_cpu[0].flatten()
        if flattened.numel() == 0:
            continue

        print(f"\nFeature: {feature_key}")
        print(f"  Shape: {attr_cpu[0].shape}")
        print(f"  Total attribution sum: {flattened.sum().item():+.6f}")
        print(f"  Mean attribution: {flattened.mean().item():+.6f}")

        k = min(top_k, flattened.numel())
        top_values, top_indices = torch.topk(flattened.abs(), k=k)
        processor = processors.get(feature_key) if processors else None
        is_continuous = torch.is_floating_point(feature_input)

        print(f"\n  Top {k} most important features:")
        for rank, (_, flat_idx) in enumerate(zip(top_values, top_indices), 1):
            attribution_value = flattened[flat_idx].item()
            coords = unravel(flat_idx.item(), attr_cpu[0].shape)

            if is_continuous:
                actual_value = feature_input[0][tuple(coords)].item()
                label = ""
                if feature_key == "labs" and len(coords) >= 1:
                    lab_idx = coords[-1]
                    if lab_idx < len(LAB_CATEGORY_NAMES):
                        label = f"{LAB_CATEGORY_NAMES[lab_idx]} "
                print(
                    f"    {rank:2d}. idx={coords} {label}value={actual_value:.4f} "
                    f"{method_name}={attribution_value:+.6f}"
                )
            else:
                token_idx = int(feature_input[0][tuple(coords)].item())
                token = decode_token(token_idx, processor, feature_key, icd_code_to_desc)
                print(
                    f"    {rank:2d}. idx={coords} token='{token}' "
                    f"{method_name}={attribution_value:+.6f}"
                )


def main():
    """Main function to run LIME analysis on MIMIC-IV StageNet model."""
    # Configure dataset location and load cached processors
    dataset = MIMIC4EHRDataset(
        #root="/home/naveen-baskaran/physionet.org/files/mimic-iv-demo/2.2/",
        #root="/Users/naveenbaskaran/data/physionet.org/files/mimic-iv-demo/2.2/",
        root="~/data/physionet.org/files/mimic-iv-demo/2.2/",
        tables=[
            "patients",
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "labevents",
        ],
    )

    # %% Setting StageNet Mortality Prediction Task
    input_processors, output_processors = load_processors("../resources/")

    sample_dataset = dataset.set_task(
        MortalityPredictionStageNetMIMIC4(),
        input_processors=input_processors,
        output_processors=output_processors,
    )
    print(f"Total samples: {len(sample_dataset)}")

    ICD_CODE_TO_DESC = load_icd_description_map(dataset.root)

    # %% Loading Pretrained StageNet Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = StageNet(
        dataset=sample_dataset,
        embedding_dim=128,
        chunk_size=128,
        levels=3,
        dropout=0.3,
    )

    state_dict = torch.load("../resources/best.ckpt", map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(model)

    # %% Preparing dataloaders
    _, _, test_data = split_by_patient(sample_dataset, [0.7, 0.1, 0.2], seed=42)
    test_loader = get_dataloader(test_data, batch_size=1, shuffle=False)


# %% Run LIME on a held-out sample
    print("\n" + "="*80)
    print("Initializing LIME Explainer")
    print("="*80)

    # Initialize LIME explainer
    lime_explainer = LimeExplainer(model)

    print("\nLIME Configuration:")
    print(f"  Use embeddings: {lime_explainer.use_embeddings}")
    print(f"  Number of samples: {lime_explainer.n_samples}")
    print(f"  Kernel width: {lime_explainer.kernel_width}")
    print(f"  Distance mode: {lime_explainer.distance_mode}")
    print(f"  Feature selection: {lime_explainer.feature_selection}")
    print(f"  Regularization alpha: {lime_explainer.alpha}")

    # Get a sample from test set
    sample_batch = next(iter(test_loader))
    sample_batch_device = move_batch_to_device(sample_batch, device)

    # Get model prediction
    with torch.no_grad():
        output = model(**sample_batch_device)
        probs = output["y_prob"]
        label_key = model.label_key
        true_label = sample_batch_device[label_key]

        # Handle binary classification (single probability output)
        if probs.shape[-1] == 1:
            prob_death = probs[0].item()
            prob_survive = 1 - prob_death
            preds = (probs > 0.5).long()
        else:
            # Multi-class classification
            preds = torch.argmax(probs, dim=-1)
            prob_survive = probs[0][0].item()
            prob_death = probs[0][1].item()

        print("\n" + "="*80)
        print("Model Prediction for Sampled Patient")
        print("="*80)
        print(f"  True label: {int(true_label.item())} {'(Deceased)' if true_label.item() == 1 else '(Survived)'}")
        print(f"  Predicted class: {int(preds.item())} {'(Deceased)' if preds.item() == 1 else '(Survived)'}")
        print(f"  Probabilities: [Survive={prob_survive:.4f}, Death={prob_death:.4f}]")

    # Compute LIME values
    print("\n" + "="*80)
    print("Computing LIME Attributions (...........)")
    print("="*80)
    print("\nLIME trains a local linear model by sampling perturbed inputs")
    print("around the example to be explained. The linear model coefficients")
    print("represent feature importance in the local neighborhood.")

    attributions = lime_explainer.attribute(**sample_batch_device, target_class_idx=1)

    print("\n" + "="*80)
    print("LIME Attribution Results")
    print("="*80)
    print("\nLIME coefficients explain the contribution of each feature to the")
    print("local linear approximation of the model's MORTALITY prediction (class 1).")
    print("Positive values increase the mortality prediction, negative values decrease it.")

    print_top_attributions(
        attributions,
        sample_batch_device,
        input_processors,
        top_k=15,
        icd_code_to_desc=ICD_CODE_TO_DESC,
        method_name="LIME"
    )

    # %% Compare different LIME configurations
    print("\n\n" + "="*80)
    print("Testing Different LIME Configurations")
    print("="*80)

    # 1. Default configuration (already computed)
    print("\n1. Default LIME (Lasso, 1000 samples, cosine distance):")
    print(f"   Total attribution (icd_codes): {attributions['icd_codes'][0].sum().item():+.6f}")
    print(f"   Total attribution (labs): {attributions['labs'][0].sum().item():+.6f}")

    # 2. Ridge regression instead of Lasso
    print("\n2. Ridge regression (L2 regularization):")
    lime_ridge = LimeExplainer(
        model,
        use_embeddings=True,
        n_samples=1000,
        feature_selection="ridge",
        alpha=0.01,
        random_seed=42,
    )
    attr_ridge = lime_ridge.attribute(**sample_batch_device, target_class_idx=1)
    print(f"   Total attribution (icd_codes): {attr_ridge['icd_codes'][0].sum().item():+.6f}")
    print(f"   Total attribution (labs): {attr_ridge['labs'][0].sum().item():+.6f}")

    # 3. Euclidean distance instead of cosine
    print("\n3. Euclidean distance kernel:")
    lime_euclidean = LimeExplainer(
        model,
        use_embeddings=True,
        n_samples=1000,
        distance_mode="euclidean",
        kernel_width=0.25,
        random_seed=42,
    )
    attr_euclidean = lime_euclidean.attribute(**sample_batch_device, target_class_idx=1)
    print(f"   Total attribution (icd_codes): {attr_euclidean['icd_codes'][0].sum().item():+.6f}")
    print(f"   Total attribution (labs): {attr_euclidean['labs'][0].sum().item():+.6f}")

    # 4. More samples for better local approximation
    print("\n4. More samples (2000) for better local approximation:")
    lime_more_samples = LimeExplainer(
        model,
        use_embeddings=True,
        n_samples=2000,
        random_seed=42,
    )
    attr_more_samples = lime_more_samples.attribute(**sample_batch_device, target_class_idx=1)
    print(f"   Total attribution (icd_codes): {attr_more_samples['icd_codes'][0].sum().item():+.6f}")
    print(f"   Total attribution (labs): {attr_more_samples['labs'][0].sum().item():+.6f}")

    print("\n" + "="*80)
    print("Comparison of Regularization Methods")
    print("="*80)
    print("\nLasso (L1) vs Ridge (L2) regularization:")
    print("  - Lasso tends to produce sparser explanations (more zeros)")
    print("  - Ridge distributes importance more evenly across features")
    print("  - Choose based on your interpretation needs:")
    print("    * Lasso: When you want to identify a few key features")
    print("    * Ridge: When you want to see contributions from all features")

    # Compare top features
    print("\nTop 5 features comparison (by absolute value):")
    for key in ['icd_codes', 'labs']:
        if key in attributions:
            flat_lasso = attributions[key][0].flatten().abs()
            flat_ridge = attr_ridge[key][0].flatten().abs()

            k = min(5, flat_lasso.numel())
            top_lasso = torch.topk(flat_lasso, k=k)
            top_ridge = torch.topk(flat_ridge, k=k)

            print(f"\n  {key}:")
            print(f"    Lasso non-zero features: {(flat_lasso > 1e-6).sum().item()}/{flat_lasso.numel()}")
            print(f"    Ridge non-zero features: {(flat_ridge > 1e-6).sum().item()}/{flat_ridge.numel()}")

    print("\n" + "="*80)
    print("LIME Analysis Complete")
    print("="*80)


if __name__ == "__main__":
    main()

# %%
