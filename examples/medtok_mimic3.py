"""MedTok vs learned code embeddings on MIMIC-III (Su et al., ICML 2025).

MedTok's claim is that swapping an EHR model's code embeddings for MedTok's
improves its predictions. This script trains the same PyHealth model twice
per seed, once with its usual learned embeddings and once with MedTok's
(:func:`pyhealth.models.apply_medtok`), and reports the test metrics as mean
(std) over seeds.

1. Load MIMIC-III and set one of PyHealth's tasks: in-hospital mortality,
   readmission, length of stay or drug recommendation (the paper's tasks).
2. Split by patient (70/10/20), with a different split for every seed.
3. For each embedding, train with early stopping on the validation AUPRC (or
   AUROC for length of stay) and score the best epoch on the test split.

Get the embeddings from https://huggingface.co/mims-harvard/MedTok
(``code2embeddings.json``, 4 GB) and convert them once to a 630 MB ``.npz``::

    python -c "from pyhealth.models import convert_medtok_embeddings as c; \\
        c('code2embeddings.json', 'medtok.npz')"

Usage::

    # Smoke test on the MIMIC-III demo bundled with the repository:
    python examples/medtok_mimic3.py --medtok medtok.npz --demo --epochs 2
    # The full comparison:
    python examples/medtok_mimic3.py --root /data/mimic-iii/1.4 \\
        --medtok medtok.npz --task mortality --seeds 0 1 2

``--embeddings tokens`` also runs MedTok's token mode, which needs
``code2tokens.json``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import MIMIC3Dataset, get_dataloader, split_by_patient
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
    LengthOfStayPredictionMIMIC3,
    MortalityPredictionMIMIC3,
    ReadmissionPredictionMIMIC3,
)
from pyhealth.trainer import Trainer

DEMO_ROOT = Path(__file__).resolve().parent.parent / "test-resources/core/mimic3demo"
CODES = {"conditions": "ICD9CM", "procedures": "ICD9PROC", "drugs": "NDC"}
TASKS = {
    "mortality": (MortalityPredictionMIMIC3, CODES, ["roc_auc", "pr_auc"]),
    "readmission": (ReadmissionPredictionMIMIC3, CODES, ["roc_auc", "pr_auc"]),
    "los": (
        LengthOfStayPredictionMIMIC3,
        CODES,
        ["roc_auc_macro_ovr", "accuracy", "f1_macro"],
    ),
    # Drug recommendation's inputs and labels are ATC level-3 codes.
    "drugrec": (
        DrugRecommendationMIMIC3,
        {**CODES, "drugs_hist": "ATC", "drugs": None},
        ["pr_auc_samples", "roc_auc_samples", "jaccard_samples"],
    ),
}
MODELS = {"transformer": Transformer, "rnn": RNN, "retain": RETAIN}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", help="MIMIC-III 1.4 folder (with ADMISSIONS.csv.gz).")
    p.add_argument("--demo", action="store_true", help="Use the bundled demo.")
    p.add_argument("--cache-dir", default=None)
    p.add_argument(
        "--medtok", help="medtok.npz or code2embeddings.json, for --embeddings medtok."
    )
    p.add_argument("--tokens", help="code2tokens.json, for --embeddings tokens.")
    p.add_argument("--task", choices=TASKS, default="mortality")
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
    p.add_argument("--output", default="output/medtok_mimic3")
    p.add_argument(
        "--device",
        default="cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )
    args = p.parse_args()
    if not args.demo and not args.root:
        p.error("pass --root or --demo")
    if "tokens" in args.embeddings and not args.tokens:
        p.error("--embeddings tokens needs --tokens code2tokens.json")
    if "medtok" in args.embeddings and not args.medtok:
        p.error("--embeddings medtok needs --medtok medtok.npz")
    return args


def load_samples(args):
    dataset = MIMIC3Dataset(
        root=str(DEMO_ROOT) if args.demo else args.root,
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        cache_dir=args.cache_dir,
    )
    return dataset.set_task(TASKS[args.task][0]())


def build_model(samples, args, embeddings, medtok):
    kwargs = {"embedding_dim": args.embedding_dim}
    if args.model == "rnn":
        kwargs["hidden_dim"] = args.embedding_dim
    model = MODELS[args.model](samples, **kwargs)
    if embeddings != "learned":
        systems = {
            k: v
            for k, v in TASKS[args.task][1].items()
            if v and k in model.feature_keys
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
    metrics = TASKS[args.task][2]
    trainer = Trainer(
        model=model,
        metrics=metrics,
        device=args.device,
        output_path=args.output,
        exp_name=f"{args.task}-{args.model}-{embeddings}-seed{seed}",
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
    systems = {k: v for k, v in TASKS[args.task][1].items() if v}
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
    name = out / f"{args.task}-{args.model}.json"
    name.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\n{args.task}, {args.model}, seeds {args.seeds}: test mean (std)")
    metrics = TASKS[args.task][2]
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
