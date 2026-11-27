"""LUDB ECG delineation with MedTsLLM: full pipeline and ablations.

Contributors: Thi Van Trinh Tran and Zexi Li.
Paper: MedTsLLM: Leveraging LLMs for Multimodal Medical Time Series Analysis.
Chan et al., MLHC 2024. https://arxiv.org/abs/2408.07773.
Data: https://physionet.org/content/ludb/1.0.1/.

What it shows:
    LUDBDataset -> ECGDelineationLUDB -> MedTsLLM -> point-wise ECG
    predictions and segmentation metrics. The task assigns background,
    P wave, QRS complex, or T wave to each supervised ECG sample.
    Ignored annotation points are excluded from loss and evaluation.

How to run:
    Offline synthetic example (NOT a scientific experiment)::

        python examples/ludb_segmentation_medtsllm.py \
            --demo --output /tmp/ludb-demo

    One real condition, using existing frozen artifacts and a local snapshot::

        python examples/ludb_segmentation_medtsllm.py \
            --manifest /project/window_manifest.csv \
            --split-manifest /project/patient_split.csv \
            --records-root /project/data \
            --mask-root /project/prepared \
            --metadata-path /project/metadata_prepared.csv \
            --cache-dir /project/cache \
            --llm-name /models/gpt2-xl/snapshot \
            --condition all_correct --seeds 0 1 2 \
            --output /project/NEW_runs/gpt2xl

    Use --condition all for nine conditions x three seeds:
    27 sequential runs per backbone.

    For Llama-2-7B, supply its local base-model snapshot through
    --llm-name and use a separate output root.

    No tests, experiments, downloads, or environment installation
    run on import.

Experimental setup:
    We evaluated nine conditions with three seeds (0, 1, 2) for each
    backbone, completing 54 LUDB experiments in total. Llama-2-7B was
    the main backbone; GPT-2 XL was the comparison.

    Backbone weights remained frozen, including in the random-
    initialization condition. The ECG adaptation and prediction
    layers were trained.

    The experiments used a fixed patient-level 80/20 train/test split
    and 512-sample windows. The original 12-lead recordings were sampled
    at 500 Hz, with each lead represented separately. We used previously
    prepared labels and annotation ignore masks, without filtering
    or downsampling.

    Training used 10 epochs, batch size 32, AdamW with learning rate
    1e-4 and weight decay 0, and gradient clipping at 1.0.

    We reported final-epoch test results, without cross-validation,
    test-based checkpoint selection, or test-based early stopping.

    Each run's summary reported 4,320 test windows, 1,761,928 supervised
    points, and 449,912 ignored points.

Ablations:
    1. Prompt components:
       No prompt; dataset only; task only; patient only; statistics only;
       or all information with correct patient context.

    2. Patient context:
       Correct patient context; mismatched patient context; or patient
       context removed, while retaining the other prompt components.

    3. Backbone initialization:
       Pretrained frozen weights versus randomly initialized frozen
       weights, using the all-information/correct-patient prompt.

    All conditions were evaluated with both backbones and three seeds.

    The condition names accepted by --condition are:

        prompt_none:    No textual prompt.
        prompt_dataset: Dataset information only.
        prompt_task:    Task information only.
        prompt_patient: Patient information only.
        prompt_stats:   Signal statistics only.
        all_correct:    All information with correct patient context.
        all_mismatch:   All information with mismatched patient context.
        all_removed:    All information with patient context removed.
        all_random:     All information with a random frozen backbone.


Results:
    Values are mean +/- sample standard deviation across three seeds,
    in percent except evaluation loss.

    F1 and point IoU are macro averages over background, P, QRS, and T.
    Segment mIoU is reported separately using our segment-matching metric.

    Llama-2-7B::

        Condition               Macro F1         Point IoU        Segment mIoU
        All / correct           86.08 +/- 0.07    76.32 +/- 0.13    61.72 +/- 1.14
        No prompt               86.75 +/- 0.01    77.27 +/- 0.01    62.58 +/- 1.00
        Dataset only            86.73 +/- 0.12    77.23 +/- 0.21    61.44 +/- 0.94
        Task only               86.94 +/- 0.10    77.54 +/- 0.16    61.47 +/- 0.88
        Patient only            85.85 +/- 0.22    76.00 +/- 0.29    60.92 +/- 0.56
        Statistics only         86.80 +/- 0.17    77.32 +/- 0.25    61.34 +/- 0.70
        All / mismatched        86.58 +/- 0.20    77.02 +/- 0.28    61.04 +/- 0.52
        All / patient removed   86.75 +/- 0.05    77.26 +/- 0.05    61.94 +/- 0.88
        All / random backbone   85.74 +/- 0.32    75.79 +/- 0.44    58.20 +/- 0.61

    GPT-2 XL::

        Condition               Macro F1         Point IoU        Segment mIoU
        All / correct           84.69 +/- 0.34    74.37 +/- 0.47    57.06 +/- 1.28
        No prompt               85.01 +/- 1.02    74.72 +/- 1.50    56.65 +/- 4.17
        Dataset only            86.44 +/- 0.14    76.79 +/- 0.24    59.61 +/- 0.68
        Task only               86.23 +/- 0.15    76.51 +/- 0.24    59.98 +/- 0.88
        Patient only            83.84 +/- 0.39    73.19 +/- 0.54    56.00 +/- 1.33
        Statistics only         86.42 +/- 0.08    76.78 +/- 0.08    59.77 +/- 0.60
        All / mismatched        85.74 +/- 0.05    75.80 +/- 0.09    58.90 +/- 0.76
        All / patient removed   86.40 +/- 0.20    76.75 +/- 0.27    59.98 +/- 0.26
        All / random backbone   81.22 +/- 0.88    69.68 +/- 1.01    44.41 +/- 1.50

    Additional metrics for the all-information/correct-patient condition::

        Metric                  Llama-2-7B          GPT-2 XL
        Accuracy (%)            88.93 +/- 0.14      87.96 +/- 0.27
        Macro precision (%)     88.43 +/- 0.96      87.38 +/- 0.77
        Macro recall (%)        84.38 +/- 0.64      82.82 +/- 0.40
        Evaluation loss         0.2973 +/- 0.0041   0.3126 +/- 0.0067

    Precision and recall were reconstructed from saved confusion matrices.
    The other metrics were read from individual summary.json files.
    Recomputed accuracy, macro F1, and macro IoU agreed with stored scores.

    Evaluation loss was the unweighted mean of cross-entropy losses over
    batches containing supervised points.

Findings:
    With all information and correct patient context, Llama-2-7B had
    higher mean macro F1, point IoU, and segment mIoU than GPT-2 XL.

    All-information prompts did not consistently outperform simpler
    prompts. Task-only prompting had the highest mean macro F1 for
    Llama-2-7B, while dataset-only prompting had the highest mean
    macro F1 for GPT-2 XL.

    Correct patient context did not improve mean macro F1 over removing
    patient context in either backbone. This finding concerns the context
    representation and experimental setup evaluated here, not the
    general clinical usefulness of patient information.

    Pretrained backbones had higher mean segment mIoU than randomly
    initialized frozen backbones: 61.72 versus 58.20 for Llama-2-7B,
    and 57.06 versus 44.41 for GPT-2 XL.

    These comparisons are descriptive results across three seeds, not
    statistical-significance or causal claims.

Comparison with the paper:
    The paper reported LUDB F1=89.89% and IoU=81.73% in Table 2.
    Its backbone comparison reported IoU=80.25% for Llama-2-7B base,
    81.73% for Llama-2-7B Chat, and 80.50% for GPT-2 XL (Table 12).
    Our Llama experiments used the base model, not Chat.

    The following comparison uses the paper's reported F1 (Table 6)
    and our mean macro F1 across three seeds::

        Prompt          Paper F1    Our Llama-2-7B    Our GPT-2 XL
        No prompt       87.40       86.75             85.01
        Dataset only    89.21       86.73             86.44
        Task only       88.79       86.94             86.23
        Patient only    89.31       85.85             83.84
        Statistics only 89.23       86.80             86.42
        All information 88.41       86.08             84.69

    Our F1 values were numerically lower. Both studies found that
    including all prompt components was not consistently best.
    Patient-only prompting performed best in the paper's table,
    whereas task-only and dataset-only prompting performed best
    in our Llama-2-7B and GPT-2 XL experiments, respectively.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from transformers import GPT2Config, GPT2Model

from pyhealth.datasets import LUDBDataset, create_sample_dataset
from pyhealth.metrics import assemble_ecg_sequences, ecg_delineation_metrics
from pyhealth.models import MedTsLLM
from pyhealth.tasks import ECGDelineationLUDB

CONDITIONS = {
    "prompt_none": ("none", "correct", False),
    "prompt_dataset": ("dataset", "correct", False),
    "prompt_task": ("task", "correct", False),
    "prompt_patient": ("patient", "correct", False),
    "prompt_stats": ("stats", "correct", False),
    "all_correct": ("all", "correct", False),
    "all_mismatch": ("all", "mismatch", False),
    "all_removed": ("all", "removed", False),
    "all_random": ("all", "correct", True),
}


class _DemoTokenizer:
    # Stable character encoding used ONLY with the tiny synthetic model.
    pad_token, eos_token = "<pad>", "<eos>"

    def __call__(self, texts, **kwargs):
        width = max(len(text) for text in texts)
        tokens = [[1 + ord(char) % 30 for char in text]
                  + [0] * (width - len(text)) for text in texts]
        return SimpleNamespace(input_ids=torch.tensor(tokens, dtype=torch.long))


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _demo_data():
    samples = []
    for pid in (1, 2):
        labels = np.tile(np.arange(4), 128)
        ignored = np.zeros(512, dtype=bool)
        ignored[:5] = True
        samples.append({
            "patient_id": str(pid), "record_id": f"{pid}:i:0",
            "signal": np.sin(np.arange(512) / 10 + pid).astype(np.float32),
            "dense_label": labels, "ignore_mask": ignored,
            "label": {"time": np.flatnonzero(~ignored).tolist(),
                      "values": labels[~ignored].tolist()},
            "patient_context": f'{{"ID":{pid}}}', "lead_index": 0,
            "start_sample": 0, "end_sample_exclusive": 512, "manifest_order": 0,
        })
    train = create_sample_dataset(
        samples=[samples[0]], input_schema=ECGDelineationLUDB.input_schema,
        output_schema=ECGDelineationLUDB.output_schema, in_memory=True,
    )
    test = create_sample_dataset(
        samples=[samples[1]], input_schema=ECGDelineationLUDB.input_schema,
        output_schema=ECGDelineationLUDB.output_schema, in_memory=True,
        input_processors=train.input_processors,
        output_processors=train.output_processors,
    )
    return train, test, {pid: f'{{"ID":{pid}}}' for pid in (1, 2)}, {}


def _on_device(dataset, device):
    # Native caches need not have physical patient order; restore historical
    # numeric-patient/source-window order before seeded GPU shuffling.
    samples = sorted(list(dataset), key=lambda row: (
        int(row["patient_id"]), int(row["manifest_order"]),
    ))
    for sample in samples:
        for key in ("signal", "dense_label", "ignore_mask"):
            sample[key] = sample[key].to(device)
        sample["label"] = tuple(value.to(device) for value in sample["label"])
    return samples


def _batches(samples, processor, contexts, batch_size, generator, shuffle):
    device = samples[0]["signal"].device
    indices = torch.arange(len(samples), device=device)
    if shuffle:
        indices = indices[torch.randperm(
            len(samples), device=device, generator=generator,
        )]
    for start in range(0, len(samples), batch_size):
        rows = [samples[index] for index in indices[start:start + batch_size].tolist()]
        batch = {key: torch.stack([row[key] for row in rows])
                 for key in ("signal", "dense_label", "ignore_mask")}
        batch["label"] = processor.collate([row["label"] for row in rows])
        batch["patient_context"] = [contexts[int(row["patient_id"])] for row in rows]
        yield batch, rows


def _evaluate(model, samples, processor, contexts, batch_size, device, seed):
    model.eval()
    true, pred, ignored, losses, parts = [], [], [], [], {}
    generator = torch.Generator(device=device).manual_seed(seed + 10_000)
    with torch.no_grad():
        for batch, rows in _batches(
            samples, processor, contexts, batch_size, generator, False,
        ):
            output = model(**batch)
            if output["supervised_points"]:
                losses.append(output["loss"].item())
            y = batch["dense_label"].cpu().numpy()
            p = output["dense_logit"].argmax(1).cpu().numpy()
            mask = batch["ignore_mask"].cpu().numpy()
            true.append(y)
            pred.append(p)
            ignored.append(mask)
            for index, row in enumerate(rows):
                key = (int(row["patient_id"]), int(row["lead_index"]))
                parts.setdefault(key, []).append((
                    int(row["start_sample"]), int(row["end_sample_exclusive"]),
                    y[index], p[index], ~mask[index],
                ))
    scores = ecg_delineation_metrics(
        np.concatenate(true), np.concatenate(pred), np.concatenate(ignored),
        sequences=assemble_ecg_sequences(parts),
    )
    # Preserve historical reporting: unweighted mean of non-empty batch CEs.
    scores["loss"] = float(np.mean(losses)) if losses else None
    scores["loss_aggregation"] = "mean_nonempty_batch_cross_entropy"
    return scores


def _run(args, train, test, base_contexts, artifacts, source_info, condition, seed):
    _seed(seed)
    prompt, context_mode, random_backbone = CONDITIONS[condition]
    contexts = dict(base_contexts)
    if context_mode == "removed":
        contexts = dict.fromkeys(contexts, "")
    elif context_mode == "mismatch":
        ids = sorted(contexts)
        # Historical intervention rotates ALL metadata IDs, including across
        # the split. Retained explicitly, not relabeled as a train-only shuffle.
        contexts = {pid: base_contexts[ids[(index + 1) % len(ids)]]
                    for index, pid in enumerate(ids)}
    if prompt in {"patient", "all"} and context_mode != "removed" and (
        not contexts or not all(contexts.values())
    ):
        raise ValueError("Patient prompts require complete non-empty metadata")
    output_dir = args.output / condition / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cpu" if args.demo else args.device)
    if args.demo:
        backbone = GPT2Model(GPT2Config(
            vocab_size=32, n_embd=8, n_layer=1, n_head=2, n_positions=1024,
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        ))
        extra = {"backbone": backbone, "tokenizer": _DemoTokenizer(),
                 "d_model": 4, "num_tokens": 4, "dropout": 0.0}
    else:
        extra = {}
    model = MedTsLLM(
        train, llm_name=args.llm_name or "synthetic-demo", prompt_mode=prompt,
        random_backbone=random_backbone, **extra,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad],
        lr=1e-4, weight_decay=0.0,
    )
    train_rows, test_rows = _on_device(train, device), _on_device(test, device)
    if not train_rows or not test_rows:
        raise ValueError("Both frozen split parts must contain samples")
    observed = {int(row["patient_id"]) for row in train_rows + test_rows}
    if not observed <= set(contexts):
        raise ValueError("Patient context mapping is incomplete")
    processor = train.output_processors["label"]
    generator = torch.Generator(device=device).manual_seed(seed)
    history = []
    for epoch in range(1, (1 if args.demo else 10) + 1):
        model.train()
        losses = []
        for batch, _ in _batches(
            train_rows, processor, contexts, args.batch_size, generator, True,
        ):
            optimizer.zero_grad(set_to_none=True)
            if batch["ignore_mask"].all():
                continue
            loss = model(**batch)["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses))
                        if losses else None})
        print(json.dumps({"condition": condition, "seed": seed, **history[-1]}),
              flush=True)
    scores = _evaluate(
        model, test_rows, processor, contexts, args.batch_size, device, seed,
    )
    provenance = {
        "condition": condition, "seed": seed, **source_info,
        "label_artifacts": artifacts, "reporting": "final_epoch_only",
        "context_intervention": "train_and_test",
        "implementation": "native_pyhealth_port_not_historical_run",
        "synthetic": args.demo,
    }
    model.save_adapters(output_dir / "final_adapters.pt", provenance=provenance)
    if random_backbone:
        # Random weights cannot be recovered from a pretrained model locator.
        # Preserve them in a full standard checkpoint, not only a seed claim.
        torch.save(model.state_dict(), output_dir / "final_full.pt")
    with (output_dir / "summary.json").open("x") as stream:
        json.dump({
            "status": "COMPLETED", "config": {**provenance, **model.adapter_config},
            "history": history, "test_metrics": scores,
            "checkpoint": "final_adapters.pt",
            "full_checkpoint": "final_full.pt" if random_backbone else None,
        }, stream, indent=2, sort_keys=True)
    print(json.dumps(scores, sort_keys=True), flush=True)


def main() -> None:
    """Run only the explicitly selected synthetic or real-data example.

    Examples:
        >>> main()  # doctest: +SKIP
    """
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true")
    source.add_argument("--manifest", type=Path)
    for name in ("split-manifest", "records-root", "mask-root", "metadata-path",
                 "cache-dir"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--llm-name")
    parser.add_argument("--condition", choices=[*CONDITIONS, "all"],
                        default="all_correct")
    parser.add_argument(
        "--seeds", type=int, choices=(0, 1, 2), nargs="+", default=[0, 1, 2],
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("batch-size must be positive; duplicate seeds are not allowed")
    if args.output.exists():
        parser.error(
            "Use a NEW output directory; existing results are never overwritten"
        )
    # Capture source files once before any data processing/model training.
    # A source checkout need not have installed distribution metadata.
    versions = {}
    for name in ("pyhealth", "torch", "transformers", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed-as-distribution"
    source_hashes = {
        item.__module__: hashlib.sha256(
            Path(inspect.getfile(item)).read_bytes(),
        ).hexdigest()
        for item in (LUDBDataset, ECGDelineationLUDB, MedTsLLM,
                     ecg_delineation_metrics)
    }
    source_hashes["example"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source_info = {"versions": versions, "source_sha256": source_hashes}
    if args.demo:
        train, test, contexts, artifacts = _demo_data()
    else:
        required = (args.split_manifest, args.records_root, args.mask_root,
                    args.metadata_path, args.cache_dir, args.llm_name)
        if not all(required):
            parser.error("Real data requires all artifact paths and a local llm-name")
        if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
            parser.error("Real training requires an available CUDA device")
        raw = LUDBDataset(
            args.manifest, args.records_root, args.mask_root, args.split_manifest,
            cache_dir=args.cache_dir, metadata_path=args.metadata_path,
        )
        task = raw.default_task
        train, test = raw.set_task(task, split=raw.manifest_split)
        contexts = task.contexts
        artifacts = {"mask_and_metadata_sha256": task.artifact_hashes,
                     "record_size_mtime": task.record_fingerprints,
                     "manifest_sha256": hashlib.sha256(
                         args.manifest.read_bytes(),
                     ).hexdigest(),
                     "split_manifest_sha256": hashlib.sha256(
                         args.split_manifest.read_bytes(),
                     ).hexdigest(),
                     "split": raw.manifest_split.to_dict()}
    conditions = list(CONDITIONS) if args.condition == "all" else [args.condition]
    for condition in conditions:
        for seed in args.seeds:
            _run(args, train, test, contexts, artifacts, source_info, condition, seed)


if __name__ == "__main__":
    main()
