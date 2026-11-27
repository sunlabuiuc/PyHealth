"""MedTok vs learned code embeddings on MIMIC-III and MIMIC-IV (Su et al., ICML 2025).

MedTok's claim is that swapping an EHR model's code embeddings for MedTok's
improves its predictions. This script trains the same PyHealth model twice
per seed, once with its usual learned embeddings and once with MedTok's
(:func:`pyhealth.models.apply_medtok`), and reports the test metrics as mean
(std) over seeds.

1. Load MIMIC-III or MIMIC-IV and set one of PyHealth's tasks: in-hospital
   mortality, readmission, length of stay or drug recommendation (the paper's
   tasks). MIMIC-IV's mortality task is left out: it drops each code's ICD
   version, and MIMIC-IV mixes ICD-9 and ICD-10 codes that cannot be told
   apart without it (``E8889`` is both ICD-9 E888.9 and ICD-10 E88.89). The
   other MIMIC-IV tasks keep it (``9_4280``, ``10_E119``), which MedTok's
   ``ICD_DX`` / ``ICD_PROC`` code systems read.
2. Split by patient (70/10/20), with a different split for every seed.
3. For each embedding, train with early stopping on the validation AUPRC (or
   AUROC for length of stay) and score the best epoch on the test split.

Get the embeddings from https://huggingface.co/mims-harvard/MedTok
(``code2embeddings.json``, 4 GB) and convert them once to a 630 MB ``.npz``::

    python -c "from pyhealth.models import convert_medtok_embeddings as c; \\
        c('code2embeddings.json', 'medtok.npz')"

Usage::

    # Smoke tests on the MIMIC-III and MIMIC-IV demos bundled with the repository:
    python examples/medtok_mimic.py --medtok medtok.npz --demo --epochs 2
    python examples/medtok_mimic.py --dataset mimic4 --task drugrec \\
        --medtok medtok.npz --demo --epochs 2
    # The full comparison:
    python examples/medtok_mimic.py --root /data/mimic-iii/1.4 \\
        --medtok medtok.npz --task mortality --seeds 0 1 2
    python examples/medtok_mimic.py --dataset mimic4 --root /data/mimic-iv/2.2 \\
        --medtok medtok.npz --task readmission --seeds 0 1 2

``--embeddings tokens`` also runs MedTok's token mode, which needs
``code2tokens.json``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import (
    MIMIC3Dataset,
    MIMIC4EHRDataset,
    get_dataloader,
    split_by_patient,
)
from pyhealth.models import (
    RETAIN,
    RNN,
    Transformer,
    apply_medtok,
    load_medtok,
    medtok_keys,
)
from pyhealth.tasks import (
    DrugRecommendationMIMIC3,
    DrugRecommendationMIMIC4,
    LengthOfStayPredictionMIMIC3,
    LengthOfStayPredictionMIMIC4,
    MortalityPredictionMIMIC3,
    ReadmissionPredictionMIMIC3,
    ReadmissionPredictionMIMIC4,
)
from pyhealth.trainer import Trainer

RESOURCES = Path(__file__).resolve().parent.parent / "test-resources/core"
DATASETS = {
    "mimic3": (MIMIC3Dataset, RESOURCES / "mimic3demo"),
    "mimic4": (MIMIC4EHRDataset, RESOURCES / "mimic4demo"),
}
BINARY = ["roc_auc", "pr_auc"]
MULTICLASS = ["roc_auc_macro_ovr", "accuracy", "f1_macro"]
MULTILABEL = ["pr_auc_samples", "roc_auc_samples", "jaccard_samples"]
CODES3 = {"conditions": "ICD9CM", "procedures": "ICD9PROC", "drugs": "NDC"}
# MIMIC-IV tasks write codes with their ICD version: "9_4280", "10_E119".
CODES4 = {"conditions": "ICD_DX", "procedures": "ICD_PROC", "drugs": "NDC"}
# Drug recommendation's inputs and labels are ATC level-3 codes.
DRUGREC = {"drugs_hist": "ATC", "drugs": None}
TASKS = {
    "mimic3": {
        "mortality": (MortalityPredictionMIMIC3, CODES3, BINARY),
        "readmission": (ReadmissionPredictionMIMIC3, CODES3, BINARY),
        "los": (LengthOfStayPredictionMIMIC3, CODES3, MULTICLASS),
        "drugrec": (DrugRecommendationMIMIC3, {**CODES3, **DRUGREC}, MULTILABEL),
    },
    "mimic4": {
        "readmission": (ReadmissionPredictionMIMIC4, CODES4, BINARY),
        "los": (LengthOfStayPredictionMIMIC4, CODES4, MULTICLASS),
        "drugrec": (DrugRecommendationMIMIC4, {**CODES4, **DRUGREC}, MULTILABEL),
    },
}
MODELS = {"transformer": Transformer, "rnn": RNN, "retain": RETAIN}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", choices=DATASETS, default="mimic3")
    p.add_argument(
        "--root", help="MIMIC-III 1.4 folder, or MIMIC-IV 2.2 (with hosp/ and icu/)."
    )
    p.add_argument("--demo", action="store_true", help="Use the bundled demo.")
    p.add_argument("--cache-dir", default=None)
    p.add_argument(
        "--medtok", help="medtok.npz or code2embeddings.json, for --embeddings medtok."
    )
    p.add_argument("--tokens", help="code2tokens.json, for --embeddings tokens.")
    p.add_argument(
        "--task", choices=["mortality", "readmission", "los", "drugrec"], default=None
    )
    p.add_argument("--model", choices=MODELS, default="transformer")
    p.add_argument(
        "--embeddings",
        nargs="+",
        choices=["learned", "medtok", "tokens"],
        default=["learned", "medtok"],
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--output", default="output/medtok_mimic")
    p.add_argument(
        "--device",
        default="cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )
    args = p.parse_args()
    args.task = args.task or (
        "mortality" if args.dataset == "mimic3" else "readmission"
    )
    if args.task not in TASKS[args.dataset]:
        p.error(
            f"--task {args.task} is not available for {args.dataset}: "
            "MortalityPredictionMIMIC4 drops each code's ICD version, and "
            "MIMIC-IV's ICD-9 and ICD-10 codes cannot be told apart without it."
        )
    args.spec = TASKS[args.dataset][args.task]
    if not args.demo and not args.root:
        p.error("pass --root or --demo")
    if "tokens" in args.embeddings and not args.tokens:
        p.error("--embeddings tokens needs --tokens code2tokens.json")
    if "medtok" in args.embeddings and not args.medtok:
        p.error("--embeddings medtok needs --medtok medtok.npz")
    return args


def load_samples(args):
    dataset_cls, demo_root = DATASETS[args.dataset]
    dataset = dataset_cls(
        root=str(demo_root) if args.demo else args.root,
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        cache_dir=args.cache_dir,
    )
    return dataset.set_task(args.spec[0]())


def build_model(samples, args, embeddings, medtok):
    kwargs = {"embedding_dim": args.embedding_dim}
    if args.model == "rnn":
        kwargs["hidden_dim"] = args.embedding_dim
    model = MODELS[args.model](samples, **kwargs)
    if embeddings != "learned":
        systems = {
            k: v for k, v in args.spec[1].items() if v and k in model.feature_keys
        }
        source = medtok["tokens" if embeddings == "tokens" else "medtok"]
        mode = "tokens" if embeddings == "tokens" else "embedding"
        coverage = apply_medtok(model, source, systems, mode=mode)
        print(f"MedTok coverage of each feature's codes: {coverage}")
    return model


def run(samples, args, seed, embeddings, medtok):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train, val, test = split_by_patient(samples, [0.7, 0.1, 0.2], seed=seed)
    loaders = [
        get_dataloader(d, batch_size=args.batch_size, shuffle=i == 0)
        for i, d in enumerate((train, val, test))
    ]
    model = build_model(samples, args, embeddings, medtok)
    metrics = args.spec[2]
    trainer = Trainer(
        model=model,
        metrics=metrics,
        device=args.device,
        output_path=args.output,
        exp_name=f"{args.dataset}-{args.task}-{args.model}-{embeddings}-seed{seed}",
    )
    trainer.train(
        train_dataloader=loaders[0],
        val_dataloader=loaders[1],
        epochs=args.epochs,
        optimizer_params={"lr": args.lr},
        monitor=metrics[0],
        monitor_criterion="max",
        load_best_model_at_last=True,
    )
    return {k: float(v) for k, v in trainer.evaluate(loaders[2]).items()}


def main():
    args = parse_args()
    samples = load_samples(args)
    print(f"{len(samples)} samples for {args.task}")
    if len(samples) == 0:
        # e.g. readmission on the synthetic MIMIC-III, whose birth dates make
        # every patient a minor, and the task drops minors.
        raise SystemExit(f"{args.task} gave no samples on this data.")
    # Read each MedTok file once, keeping only the codes this task uses.
    systems = {k: v for k, v in args.spec[1].items() if v}
    wanted = {
        key
        for feature, system in systems.items()
        if feature in samples.input_processors
        for code in getattr(samples.input_processors[feature], "code_vocab", {})
        for key in medtok_keys(code, system)
    }
    medtok = {"medtok": None, "tokens": None}
    if "medtok" in args.embeddings:
        medtok["medtok"] = load_medtok(args.medtok, wanted)
    if "tokens" in args.embeddings:
        medtok["tokens"] = load_medtok(args.tokens, wanted)

    results = {e: [] for e in args.embeddings}
    for seed in args.seeds:
        for embeddings in args.embeddings:
            scores = run(samples, args, seed, embeddings, medtok)
            results[embeddings].append(scores)
            print(f"seed {seed} {embeddings}: {json.dumps(scores)}", flush=True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    name = out / f"{args.dataset}-{args.task}-{args.model}.json"
    config = {k: v for k, v in vars(args).items() if k != "spec"}
    name.write_text(json.dumps({"args": config, "results": results}, indent=2))
    print(
        f"\n{args.dataset} {args.task}, {args.model}, seeds {args.seeds}: "
        "test mean (std)"
    )
    metrics = args.spec[2]
    print(f"{'embeddings':<10}" + "".join(f"{m:>22}" for m in metrics))
    for embeddings, runs in results.items():
        cells = []
        for m in metrics:
            vals = np.array([r[m] for r in runs])
            cells.append(f"{vals.mean():.4f} ({vals.std():.4f})")
        print(f"{embeddings:<10}" + "".join(f"{c:>22}" for c in cells))
    print(f"Saved to {name}")


if __name__ == "__main__":
    main()
