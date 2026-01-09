# %% Loading MIMIC-IV dataset
from pathlib import Path

import polars as pl
import torch

from pyhealth.datasets import (
    MIMIC4EHRDataset,
    get_dataloader,
    load_processors,
    split_by_patient,
)
from pyhealth.interpret.methods import ShapExplainer
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
):
    """Print top-k most important features from SHAP attributions."""
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
                    f"SHAP={attribution_value:+.6f}"
                )
            else:
                token_idx = int(feature_input[0][tuple(coords)].item())
                token = decode_token(token_idx, processor, feature_key, icd_code_to_desc)
                print(
                    f"    {rank:2d}. idx={coords} token='{token}' "
                    f"SHAP={attribution_value:+.6f}"
                )


def main():
    """Main function to run SHAP analysis on MIMIC-IV StageNet model."""
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
        cache_dir="~/.cache/pyhealth/mimic4_stagenet_mortality",
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


# %% Run SHAP on a held-out sample
    print("\n" + "="*80)
    print("Initializing SHAP Explainer")
    print("="*80)

    # Initialize SHAP explainer (Kernel SHAP))
    shap_explainer = ShapExplainer(model)

    print("\nSHAP Configuration:")
    print(f"  Use embeddings: {shap_explainer.use_embeddings}")
    print(f"  Background samples: {shap_explainer.n_background_samples}")
    print(f"  Max coalitions: {shap_explainer.max_coalitions}")
    print(f"  Regularization: {shap_explainer.regularization}")

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

    # Compute SHAP values
    print("\n" + "="*80)
    print("Computing SHAP Attributions (this may take a minute...)")
    print("="*80)

    attributions = shap_explainer.attribute(**sample_batch_device, target_class_idx=1)

    print("\n" + "="*80)
    print("SHAP Attribution Results")
    print("="*80)
    print("\nSHAP values explain the contribution of each feature to the model's")
    print("prediction of MORTALITY (class 1). Positive values increase the")
    print("mortality prediction, negative values decrease it.")

    print_top_attributions(attributions, sample_batch_device, input_processors, top_k=15, icd_code_to_desc=ICD_CODE_TO_DESC)

    # %% Compare different baseline strategies
    print("\n\n" + "="*80)
    print("Testing Different Baseline Strategies")
    print("="*80)

    # 1. Automatic baseline (default)
    print("\n1. Automatic baseline generation (recommended):")
    attr_auto = shap_explainer.attribute(**sample_batch_device, target_class_idx=1)
    print(f"   Total attribution (icd_codes): {attr_auto['icd_codes'][0].sum().item():+.6f}")
    print(f"   Total attribution (labs): {attr_auto['labs'][0].sum().item():+.6f}")

    # Note: Custom baselines for discrete features (like ICD codes) require careful
    # construction to avoid invalid sequences. The automatic baseline generation
    # handles this by sampling from the observed data distribution.

    # %% Test callable interface
    print("\n" + "="*80)
    print("Testing Callable Interface")
    print("="*80)

    # Both methods should produce identical results (due to random_seed)
    attr_from_attribute = shap_explainer.attribute(**sample_batch_device, target_class_idx=1)
    attr_from_call = shap_explainer(**sample_batch_device, target_class_idx=1)

    print("\nVerifying that explainer(**data) and explainer.attribute(**data) produce")
    print("identical results when random_seed is set...")

    all_close = True
    for key in attr_from_attribute.keys():
        if not torch.allclose(attr_from_attribute[key], attr_from_call[key], atol=1e-6):
            all_close = False
            print(f"  ❌ {key}: Results differ!")
        else:
            print(f"  ✓ {key}: Results match")

    if all_close:
        print("\n✓ All attributions match! Callable interface works correctly.")
    else:
        print("\n❌ Some attributions differ. Check random seed configuration.")

    print("\n" + "="*80)
    print("SHAP Analysis Complete")
    print("="*80)


if __name__ == "__main__":
    main()

# %%
