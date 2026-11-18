"""Native LUDB task/model example, without implicit downloads or submission.

Contributors: Thi Van Trinh Tran and Zexi Li.
Paper: MedTsLLM, https://arxiv.org/abs/2408.07773.

Offline synthetic example (NOT a scientific experiment)::

    python examples/ludb_segmentation_medtsllm.py --demo --output /tmp/ludb-demo

One real condition, using existing frozen artifacts and a local snapshot::

    python examples/ludb_segmentation_medtsllm.py \
      --manifest /project/window_manifest.csv \
      --split-manifest /project/patient_split.csv --records-root /project/data \
      --mask-root /project/prepared --metadata-path /project/metadata_prepared.csv \
      --cache-dir /project/cache --llm-name /models/gpt2-xl/snapshot \
      --condition all_correct --seeds 0 1 2 --output /project/NEW_runs/gpt2xl

Use --condition all for nine conditions x three seeds (27 sequential runs).
Use a different output root with the Llama-2-7B base snapshot for its 27 runs.
This example does not submit HPC jobs or run both backbones automatically.
Real runs require CUDA. WFDB reading/cache building are CPU I/O operations;
model forward/backward and stored numeric windows use the requested GPU.
Test evaluation is final-epoch only, never used for checkpoint selection.
No tests, experiments, downloads, or environment installation run on import.
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
