# %% Loading MIMIC-IV dataset
import torch

from pyhealth.datasets import (
    MIMIC4EHRDataset,
    get_dataloader,
    load_processors,
    split_by_patient,
)
from pyhealth.interpret.methods import DeepLift
from pyhealth.models import StageNet
from pyhealth.tasks import MortalityPredictionStageNetMIMIC4

# Configure dataset location and load cached processors
dataset = MIMIC4EHRDataset(
    root="/home/logic/physionet.org/files/mimic-iv-demo/2.2/",
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


LAB_CATEGORY_NAMES = [
    "Sodium",
    "Potassium",
    "Chloride",
    "Bicarbonate",
    "Glucose",
    "Calcium",
    "Magnesium",
    "Anion Gap",
    "Osmolality",
    "Phosphate",
]


def decode_token(idx: int, processor):
    if processor is None or not hasattr(processor, "code_vocab"):
        return str(idx)
    reverse_vocab = {index: token for token, index in processor.code_vocab.items()}
    return reverse_vocab.get(idx, f"<UNK:{idx}>")


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
                token = decode_token(token_idx, processor)
                print(
                    f"  {rank:2d}. idx={coords} token='{token}' "
                    f"attr={attribution_value:+.6f}"
                )


# %% Run DeepLIFT on a held-out sample
deeplift = DeepLift(model)

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

attributions = deeplift.attribute(**sample_batch_device)
print_top_attributions(attributions, sample_batch_device, input_processors, top_k=10)

# %%
