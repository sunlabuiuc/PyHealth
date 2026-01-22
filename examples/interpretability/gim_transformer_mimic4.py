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
from pyhealth.models import Transformer
from pyhealth.tasks import MortalityPredictionMIMIC4


def maybe_load_processors(resource_dir: str, task):
    """Load cached processors if they match the current task schema."""

    try:
        input_processors, output_processors = load_processors(resource_dir)
    except Exception as exc:
        print(f"Falling back to rebuilding processors: {exc}")
        return None, None

    expected_inputs = set(task.input_schema.keys())
    expected_outputs = set(task.output_schema.keys())
    if set(input_processors.keys()) != expected_inputs:
        print(
            "Cached input processors do not match MortalityPredictionMIMIC4 schema; rebuilding."
        )
        return None, None
    if set(output_processors.keys()) != expected_outputs:
        print(
            "Cached output processors do not match MortalityPredictionMIMIC4 schema; rebuilding."
        )
        return None, None
    return input_processors, output_processors


# Configure dataset location and optionally load cached processors
dataset = MIMIC4EHRDataset(
    root="/home/logic/physionet.org/files/mimic-iv-demo/2.2/",
    tables=[
        "patients",
        "admissions",
        "diagnoses_icd",
        "procedures_icd",
        "prescriptions",
    ],
)

task = MortalityPredictionMIMIC4()
input_processors, output_processors = maybe_load_processors("../resources/", task)

sample_dataset = dataset.set_task(
    task,
    cache_dir="~/.cache/pyhealth/mimic4_transformer_mortality",
    input_processors=input_processors,
    output_processors=output_processors,
)
print(f"Total samples: {len(sample_dataset)}")


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


ICD_CODE_TO_DESC = load_icd_description_map(dataset.root)

# %% Loading Pretrained Transformer model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Transformer(
    dataset=sample_dataset,
    embedding_dim=128,
    heads=2,
    dropout=0.2,
    num_layers=2,
)

ckpt_path = Path("../resources/transformer_best.ckpt")
if not ckpt_path.exists():
    raise FileNotFoundError(
        f"Missing pretrained weights at {ckpt_path}. "
        "Train the Transformer model and place the checkpoint in ../resources/."
    )
state_dict = torch.load(str(ckpt_path), map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print(model)

# %% Preparing dataloaders
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


def decode_token(idx: int, processor, feature_key: str):
    if processor is None or not hasattr(processor, "code_vocab"):
        return str(idx)
    reverse_vocab = {index: token for token, index in processor.code_vocab.items()}
    token = reverse_vocab.get(idx, f"<UNK:{idx}>")

    if feature_key in {"conditions", "procedures"} and token not in {"<unk>", "<pad>"}:
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
                print(
                    f"  {rank:2d}. idx={coords} value={actual_value:.4f} "
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
gim = GIM(model, temperature=2.0)
processors_for_display = sample_dataset.input_processors

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
print_top_attributions(attributions, sample_batch_device, processors_for_display, top_k=10)

# %%
