"""DearLLM mortality prediction on MIMIC-III, end to end.

Paper: Yongxin Xu et al. DearLLM: Enhancing Personalized Healthcare via
Large Language Models-Deduced Feature Correlations. AAAI 2025.
https://doi.org/10.1609/aaai.v39i1.32079

Pipeline:
    1. Load MIMIC-III diagnoses and collect each patient's visit history.
    2. Split patients 8:1:1 (stratified on the label) and compute IPF on the
       training patients.
    3. (Optional, needs an LLM) Score every ordered pair of graph codes with
       the conditional perplexity of the LLM (paper Eq. 2) and save it.
    4. Normalise perplexities into edge weights with bounds from the
       training patients (paper Eq. 3).
    5. Build the DearLLM samples, train with a smaller learning rate for the
       GCN layers, pick the best epoch on validation PR-AUC and report test
       PR-AUC / ROC-AUC / F1 over several seeds.

Usage:
    # Smoke run on the 100-patient demo shipped with PyHealth, no LLM
    # (every graph is edgeless). Runs on a laptop CPU in about a minute.
    python examples/mortality_prediction/dearllm_mimic3.py

    # Same, with LLM-deduced edges from a small local model (laptop-sized).
    python examples/mortality_prediction/dearllm_mimic3.py \\
        --llm Qwen/Qwen2.5-0.5B-Instruct --graphs dearllm_graphs.json

    # Paper setting (credentialed MIMIC-III, a GPU and HuatuoGPT-II 7B).
    python examples/mortality_prediction/dearllm_mimic3.py \\
        --root /path/to/mimic-iii/1.4 --llm FreedomIntelligence/HuatuoGPT2-7B \\
        --graphs mimic3_graphs.json --epochs 100 --seeds 1 2 3 4 5

The demo has only 14 eligible patients, so its metrics are noise: the demo
run only shows that the pipeline works. It is not a reproduction.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from pyhealth.datasets import MIMIC3Dataset, get_dataloader
from pyhealth.metrics import binary_metrics_fn
from pyhealth.models import DearLLM
from pyhealth.tasks import DearLLMMortalityPredictionMIMIC3
from pyhealth.tasks.dearllm_mortality import compute_ipf, normalize_perplexity_graphs

DEMO_ROOT = Path(__file__).resolve().parents[2] / "test-resources/core/mimic3demo"

# Prompt from the paper (Fig. 2) and the official code: a description of the
# visit history ("Personalized Visit History Constrained Prompt") followed by
# the trigger "<code i> is related to"; the LLM then scores "<code j>".
PROMPT_HEADER = (
    "The following data consists of a sequence of hospital visits for a single "
    "patient. Each visit lists the diagnosed diseases, and the number of days "
    "since the last visit.\n"
)
TRIGGER = "Based on the patient's medical history, {} is related to "


# ----------------------------------------------------------------------------
# Step 3 (optional): LLM-deduced edges
# ----------------------------------------------------------------------------
def code_namer():
    """Returns a function mapping an ICD-9 code to its name (or the code)."""
    try:
        from pyhealth.medcode import InnerMap

        icd9 = InnerMap.load("ICD9CM")
    except OSError as err:  # e.g. offline: fall back to raw codes
        print(f"Could not load ICD-9 names ({err}); using raw codes.")
        return str

    def name(code):
        try:
            return icd9.lookup(code)
        except KeyError:  # code not in the ICD-9 table
            return code

    return name


def visit_history_text(history, name, num_visits):
    """Formats the most recent visits as in the paper's prompt."""
    visits, times = history["visits"], history["timestamps"]
    first = max(len(visits) - num_visits, 0)  # paper: 5 most recent visits
    lines = []
    for i in range(first, len(visits)):
        if i == 0:
            gap = "First visit"
        else:
            gap = f"Days since last visit: {(times[i] - times[i - 1]).days}"
        lines.append(f"Visit {i + 1} ({gap}):")
        lines.extend(f"  - {name(code)}" for code in visits[i])
    return PROMPT_HEADER + "\n".join(lines) + "\n"


@torch.no_grad()
def conditional_perplexity(model, tokenizer, prompt, target):
    """exp(mean NLL of the target tokens given the prompt), paper Eq. 2."""
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    full_ids = tokenizer(prompt + target, return_tensors="pt").input_ids
    labels = full_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100  # only score the target
    # Keep the end of the text if it is longer than the model's context
    # (as the official code does); the target is always at the end.
    max_len = getattr(model.config, "max_position_embeddings", None)
    if max_len and full_ids.shape[1] > max_len:
        full_ids, labels = full_ids[:, -max_len:], labels[:, -max_len:]
    if (labels != -100).sum() == 0:
        return float("inf")
    loss = model(full_ids.to(model.device), labels=labels.to(model.device)).loss
    return math.exp(loss.item())


def build_perplexity_graphs(histories, task, llm_name, max_codes):
    """Scores all ordered code pairs of every patient graph with an LLM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype="auto")
    model.eval()
    name = code_namer()

    graphs = {}
    for n, (pid, history) in enumerate(histories.items()):
        nodes = []
        for visit in history["visits"][-task.graph_visits :]:
            nodes.extend(c for c in visit if c not in nodes)
        nodes = nodes[:max_codes]  # the official client caps this at 10
        context = visit_history_text(history, name, task.graph_visits)
        graphs[pid] = [
            (
                a,
                b,
                conditional_perplexity(
                    model, tokenizer, context + TRIGGER.format(name(a)), name(b)
                ),
            )
            for a in nodes
            for b in nodes
            if a != b
        ]
        print(f"  graph {n + 1}/{len(histories)}: {len(graphs[pid])} edges")
    return graphs


# ----------------------------------------------------------------------------
# Training and evaluation
# ----------------------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(**batch)
            y_true.append(out["y_true"].cpu().numpy())
            y_prob.append(out["y_prob"].cpu().numpy())
    y_true = np.concatenate(y_true).ravel()
    y_prob = np.concatenate(y_prob).ravel()
    if len(np.unique(y_true)) < 2:  # AUCs are undefined with one class
        return {"pr_auc": float("nan"), "roc_auc": float("nan"), "f1": float("nan")}
    return binary_metrics_fn(y_true, y_prob, metrics=["pr_auc", "roc_auc", "f1"])


def train_one_seed(samples, splits, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = samples.subset(splits["train"])
    loaders = {
        "train": get_dataloader(train, batch_size=args.batch_size, shuffle=True),
        "val": get_dataloader(samples.subset(splits["val"]), args.batch_size),
        "test": get_dataloader(samples.subset(splits["test"]), args.batch_size),
    }
    model = DearLLM(samples, backbone=args.backbone)
    optimizer = torch.optim.Adam(
        model.parameter_groups(lr=args.lr, graph_lr=args.graph_lr, weight_decay=1e-4)
    )
    best_score, best_state = -float("inf"), None
    for epoch in range(args.epochs):
        model.train()
        for batch in loaders["train"]:
            optimizer.zero_grad()
            model(**batch)["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        val = evaluate(model, loaders["val"])
        score = val["pr_auc"] if not math.isnan(val["pr_auc"]) else -epoch
        if score > best_score:
            best_score = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return evaluate(model, loaders["test"])


def split_patients(histories, seed):
    """Stratified 8:1:1 patient split (falls back to unstratified if tiny)."""
    pids = sorted(histories)
    labels = [histories[p]["mortality"] for p in pids]
    try:
        rest, test = train_test_split(
            pids, test_size=0.1, stratify=labels, random_state=seed
        )
        rest_labels = [histories[p]["mortality"] for p in rest]
        train, val = train_test_split(
            rest, test_size=1 / 9, stratify=rest_labels, random_state=seed
        )
    except ValueError:
        rest, test = train_test_split(pids, test_size=0.1, random_state=seed)
        train, val = train_test_split(rest, test_size=1 / 9, random_state=seed)
    return {"train": set(train), "val": set(val), "test": set(test)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=str(DEMO_ROOT), help="MIMIC-III root")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--graphs", default=None, help="JSON file of perplexities")
    parser.add_argument("--llm", default=None, help="Hugging Face causal LM")
    parser.add_argument("--max-graph-codes", type=int, default=10)
    parser.add_argument("--backbone", default="gru")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--graph-lr", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    # 1. Visit histories.
    dataset = MIMIC3Dataset(
        root=args.root, tables=["diagnoses_icd"], cache_dir=args.cache_dir
    )
    plain_task = DearLLMMortalityPredictionMIMIC3()
    histories = {}
    for patient in dataset.iter_patients():
        history = plain_task.extract_history(patient)
        if history is not None:
            histories[str(patient.patient_id)] = history
    positives = sum(h["mortality"] for h in histories.values())
    print(f"{len(histories)} patients with >= 2 admissions, {positives} positive")

    # 2. Split and IPF on training patients only.
    splits = split_patients(histories, seed=0)
    ipf = compute_ipf(
        {
            pid: [c for visit in histories[pid]["visits"] for c in visit]
            for pid in splits["train"]
        }
    )

    # 3. LLM perplexities (loaded from --graphs, built with --llm, or none).
    raw_graphs = {}
    if args.graphs and os.path.exists(args.graphs):
        with open(args.graphs) as f:
            raw_graphs = json.load(f)
        print(f"Loaded perplexity graphs from {args.graphs}")
    elif args.llm:
        print(f"Scoring code pairs with {args.llm} (slow without a GPU)")
        raw_graphs = build_perplexity_graphs(
            histories, plain_task, args.llm, args.max_graph_codes
        )
        if args.graphs:
            with open(args.graphs, "w") as f:
                json.dump(raw_graphs, f)
    else:
        print("No --graphs or --llm: graphs have no edges (DearLLM without prior).")

    # 4. Normalise with bounds from the training patients.
    graphs = {}
    if raw_graphs:
        train_graphs = {p: e for p, e in raw_graphs.items() if p in splits["train"]}
        graphs, bounds = normalize_perplexity_graphs(train_graphs)
        others = {p: e for p, e in raw_graphs.items() if p not in splits["train"]}
        graphs.update(normalize_perplexity_graphs(others, bounds=bounds)[0])
        print(f"Perplexity bounds (10th / 90th percentile): {bounds}")

    # 5. Samples, training, evaluation.
    task = DearLLMMortalityPredictionMIMIC3(graphs=graphs, ipf=ipf)
    samples = dataset.set_task(task)
    pids = [str(samples[i]["patient_id"]) for i in range(len(samples))]
    index = {
        name: [i for i, pid in enumerate(pids) if pid in members]
        for name, members in splits.items()
    }
    print({name: len(ids) for name, ids in index.items()})

    results = [train_one_seed(samples, index, args, seed) for seed in args.seeds]
    for metric in ("pr_auc", "roc_auc", "f1"):
        values = [r[metric] for r in results]
        print(f"test {metric}: {np.nanmean(values):.4f} ({np.nanstd(values):.4f})")


if __name__ == "__main__":
    main()
