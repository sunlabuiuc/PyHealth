"""GIM (Gradient Interaction Modifications) attributions for a StageNet on MIMIC-IV.

Explains one held-out patient's ``MortalityPredictionStageNetMIMIC4``
prediction with ``pyhealth.interpret.methods.GIM`` and prints the top-10
attributed ICD codes and lab values. Written as ``# %%`` cells, so it can also
be stepped through in VS Code / Jupyter.

Data: the open-access MIMIC-IV demo
(https://physionet.org/content/mimic-iv-demo/2.2/, no credentials) or full
MIMIC-IV; a copy of the demo ships in ``test-resources/core/mimic4demo``. Set
MIMIC4_ROOT below. RESOURCES_DIR may hold a trained model
(``input_processors.pkl``, ``output_processors.pkl``, ``best.ckpt``); if the
checkpoint is missing, the model is trained for 5 epochs.

Run from this folder (CPU is fine; about a minute on the demo):
    cd examples/interpretability && python gim_stagenet_mimic4.py
"""

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
from pyhealth.interpret.methods import GIM
from pyhealth.models import StageNet
from pyhealth.tasks import MortalityPredictionStageNetMIMIC4
from pyhealth.trainer import Trainer

# Paths: edit before running.
MIMIC4_ROOT = "~/physionet.org/files/mimic-iv-demo/2.2/"
RESOURCES_DIR = Path("../resources/")  # optional pre-trained model

# Configure dataset location and load cached processors
if __name__ == "__main__":  # set_task spawns workers that re-import this file
    dataset = MIMIC4EHRDataset(
        root=MIMIC4_ROOT,
        tables=[
            "patients",
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "labevents",
        ],
    )

# %% Setting StageNet Mortality Prediction Task
if __name__ == "__main__":  # set_task spawns workers that re-import this file
    input_processors = output_processors = None
    if (RESOURCES_DIR / "input_processors.pkl").exists():
        input_processors, output_processors = load_processors(str(RESOURCES_DIR))

    sample_dataset = dataset.set_task(
        MortalityPredictionStageNetMIMIC4(),
        input_processors=input_processors,
        output_processors=output_processors,
    )
    print(f"Total samples: {len(sample_dataset)}")
    input_processors = sample_dataset.input_processors


def load_icd_description_map(dataset_root: str) -> dict:
    """Load ICD code → description mappings from reference tables."""
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


if __name__ == "__main__":  # set_task spawns workers that re-import this file
    ICD_CODE_TO_DESC = load_icd_description_map(dataset.root)

# %% Loading Pretrained StageNet model
if __name__ == "__main__":  # set_task spawns workers that re-import this file
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = StageNet(
        dataset=sample_dataset,
        embedding_dim=128,
        chunk_size=128,
        levels=3,
        dropout=0.3,
    )

    ckpt_path = RESOURCES_DIR / "best.ckpt"
    if ckpt_path.exists():
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"No checkpoint at {ckpt_path}; training the model for 5 epochs.")
        train_data, _, _ = split_by_patient(sample_dataset, [0.7, 0.1, 0.2], seed=42)
        Trainer(model=model, device=str(device), enable_logging=False).train(
            train_dataloader=get_dataloader(train_data, batch_size=32, shuffle=True),
            epochs=5,
        )
    model = model.to(device)
    model.eval()
    print(model)

# %% Preparing dataloaders
if __name__ == "__main__":  # set_task spawns workers that re-import this file
    _, _, test_data = split_by_patient(sample_dataset, [0.7, 0.1, 0.2], seed=42)
    test_loader = get_dataloader(test_data, batch_size=1, shuffle=False)


def move_batch_to_device(batch, target_device):
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(target_device)
        elif isinstance(value, tuple):
            moved[key] = tuple(v.to(target_device) for v in value)
        else:
            moved[key] = value
    return moved


LAB_CATEGORY_NAMES = MortalityPredictionStageNetMIMIC4.LAB_CATEGORY_NAMES


def decode_token(idx: int, processor, feature_key: str):
    if processor is None or not hasattr(processor, "code_vocab"):
        return str(idx)
    reverse_vocab = {index: token for token, index in processor.code_vocab.items()}
    token = reverse_vocab.get(idx, f"<UNK:{idx}>")

    if feature_key == "icd_codes" and token not in {"<unk>", "<pad>"}:
        desc = ICD_CODE_TO_DESC.get(token)
        if desc:
            return f"{token}: {desc}"

    return token


def unravel(flat_index: int, shape: torch.Size):
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
):
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
        k = min(top_k, flattened.numel())
        top_values, top_indices = torch.topk(flattened.abs(), k=k)
        processor = processors.get(feature_key) if processors else None
        is_continuous = torch.is_floating_point(feature_input)

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
                    f"  {rank:2d}. idx={coords} {label}value={actual_value:.4f} "
                    f"attr={attribution_value:+.6f}"
                )
            else:
                token_idx = int(feature_input[0][tuple(coords)].item())
                token = decode_token(token_idx, processor, feature_key)
                print(
                    f"  {rank:2d}. idx={coords} token='{token}' "
                    f"attr={attribution_value:+.6f}"
                )


# %% Run GIM on a held-out sample
if __name__ == "__main__":  # set_task spawns workers that re-import this file
    gim = GIM(model, temperature=2.0)

    sample_batch = next(iter(test_loader))
    sample_batch_device = move_batch_to_device(sample_batch, device)

    with torch.no_grad():
        output = model(**sample_batch_device)
        probs = output["y_prob"]
        preds = torch.argmax(probs, dim=-1)
        label_key = model.label_key
        true_label = sample_batch_device[label_key]

        print("\nModel prediction for the sampled patient:")
        print(f"  True label: {int(true_label.item())}")
        print(f"  Predicted class: {int(preds.item())}")
        print(f"  Probabilities: {probs[0].cpu().numpy()}")

    attributions = gim.attribute(**sample_batch_device)
    print_top_attributions(
        attributions, sample_batch_device, input_processors, top_k=10
    )

# %%
