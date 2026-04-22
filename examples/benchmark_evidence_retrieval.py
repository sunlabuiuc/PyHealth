"""Reproducible benchmark runner for LLM-based evidence retrieval.

Runs all three ablations against both the offline :class:`StubLLMBackend`
and a real LLM backend (default: OpenAI ``gpt-4o-mini``), then writes:

- ``examples/results/benchmark_results.json`` — raw numbers for every
  configuration, suitable for re-plotting.
- ``examples/results/benchmark_results.md`` — human-readable summary
  tables embedded in the PR description.
- ``examples/results/*.png`` — side-by-side bar charts comparing the
  offline stub and the real LLM for each ablation.

Paper:
    M. Ahsan et al. "Retrieving Evidence from EHRs with LLMs:
    Possibilities and Challenges." Proceedings of Machine Learning
    Research, 2024.

Paper link:
    https://proceedings.mlr.press/v248/ahsan24a.html

Usage:
    # Run both stub and OpenAI backends, emit JSON + markdown + PNGs:
    python examples/benchmark_evidence_retrieval.py \\
        --llm-backend openai --model gpt-4o-mini

    # Offline-only (no OpenAI call made, charts still produced):
    python examples/benchmark_evidence_retrieval.py --llm-backend stub

Author:
    Arnab Karmakar (arnabk3@illinois.edu)
"""
import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

def _load_env() -> None:
    """Load a ``.env`` file from the repo root.

    Uses ``python-dotenv`` when installed, otherwise falls back to a
    minimal ``KEY=value`` parser so the script works without any
    optional dependency. Never overwrites existing env vars.
    """
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
        return
    except ImportError:
        pass
    here = os.path.abspath(os.path.dirname(__file__))
    for _ in range(4):
        candidate = os.path.join(here, ".env")
        if os.path.isfile(candidate):
            try:
                with open(candidate, "r") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip()
                        if (
                            len(value) >= 2
                            and value[0] == value[-1]
                            and value[0] in ("'", '"')
                        ):
                            value = value[1:-1]
                        if key and key not in os.environ:
                            os.environ[key] = value
            except OSError:
                pass
            return
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent


_load_env()

from pyhealth.datasets import SyntheticEHRNotesDataset
from pyhealth.models import (
    CBERTLiteRetriever,
    LLMEvidenceRetriever,
    LLMRetrieverConfig,
    StubLLMBackend,
)
from pyhealth.tasks import EvidenceRetrievalMIMIC3


RESULTS_DIR = Path(__file__).parent / "results"


# ----------------------------------------------------------------------
# Pipeline helpers
# ----------------------------------------------------------------------
def _metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    """Return acc / precision / recall / FP count for a binary run."""
    tp = sum(1 for p, t in zip(predictions, labels) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(predictions, labels) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(predictions, labels) if p == 0 and t == 1)
    tn = sum(1 for p, t in zip(predictions, labels) if p == 0 and t == 0)
    total = tp + fp + fn + tn
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "fp": fp,
    }


def _load_data() -> Dict[str, List]:
    """Materialise the synthetic corpus into parallel lists."""
    tmp_root = tempfile.mkdtemp(prefix="pyhealth_bench_data_")
    tmp_cache = tempfile.mkdtemp(prefix="pyhealth_bench_cache_")
    dataset = SyntheticEHRNotesDataset(root=tmp_root, cache_dir=tmp_cache)
    samples = dataset.set_task(EvidenceRetrievalMIMIC3())
    try:
        out = {
            "note_text": [],
            "condition": [],
            "note_id": [],
            "is_positive": [],
        }
        for sample in samples:
            out["note_text"].append(sample["note_text"])
            out["condition"].append(sample["condition"])
            out["note_id"].append(sample.get("note_id", ""))
            out["is_positive"].append(int(sample["is_positive"]))
        return out
    finally:
        samples.close()


def _build_backend_factory(
    name: str, model: str
) -> Callable[[], object]:
    """Return a zero-arg factory producing a fresh backend instance."""
    if name == "stub":
        return StubLLMBackend
    if name == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY not set. Add it to .env or export it."
            )
        from pyhealth.models import OpenAIBackend

        return lambda: OpenAIBackend(model=model)
    raise ValueError(f"Unknown backend: {name}")


# ----------------------------------------------------------------------
# Individual ablations
# ----------------------------------------------------------------------
def _run_prompt_style_ablation(
    data: Dict[str, List],
    backend_factory: Callable[[], object],
) -> Dict[str, Any]:
    """Compare sequential vs single-prompt styles on one backend."""
    out: Dict[str, Any] = {}
    for style in ("sequential", "single"):
        retriever = LLMEvidenceRetriever(
            backend=backend_factory(),
            config=LLMRetrieverConfig(prompt_style=style),
        )
        snippets = retriever.retrieve_evidence_batch(
            data["note_text"], data["condition"], data["note_id"]
        )
        preds = [1 if s.decision == "yes" else 0 for s in snippets]
        m = _metrics(preds, data["is_positive"])
        m["generated_count"] = sum(1 for s in snippets if s.is_generated)
        m["total"] = len(snippets)
        out[style] = m
    return out


def _run_llm_vs_ir(
    data: Dict[str, List],
    backend_factory: Callable[[], object],
) -> Dict[str, Any]:
    """LLM (sequential) vs CBERT-lite IR baseline."""
    llm = LLMEvidenceRetriever(
        backend=backend_factory(),
        config=LLMRetrieverConfig(prompt_style="sequential"),
    )
    llm_snips = llm.retrieve_evidence_batch(
        data["note_text"], data["condition"], data["note_id"]
    )
    llm_preds = [1 if s.decision == "yes" else 0 for s in llm_snips]

    baseline = CBERTLiteRetriever(top_k=2)
    baseline_outputs = baseline(
        note_text=data["note_text"],
        condition=data["condition"],
        note_id=data["note_id"],
    )
    baseline_probs = baseline_outputs["y_prob"].squeeze(-1).tolist()
    baseline_preds = [1 if p >= 0.5 else 0 for p in baseline_probs]

    return {
        "llm_sequential": _metrics(llm_preds, data["is_positive"]),
        "cbert_lite": _metrics(baseline_preds, data["is_positive"]),
    }


def _run_note_budget_sweep(
    data: Dict[str, List],
    backend_factory: Callable[[], object],
    budgets: List[int] = (80, 160, 320, 4000),
) -> Dict[str, Any]:
    """max_note_chars sweep on one backend."""
    out: Dict[str, Any] = {}
    for budget in budgets:
        retriever = LLMEvidenceRetriever(
            backend=backend_factory(),
            config=LLMRetrieverConfig(
                prompt_style="sequential", max_note_chars=budget
            ),
        )
        snippets = retriever.retrieve_evidence_batch(
            data["note_text"], data["condition"], data["note_id"]
        )
        preds = [1 if s.decision == "yes" else 0 for s in snippets]
        out[str(budget)] = _metrics(preds, data["is_positive"])
    return out


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def _write_markdown_report(
    results: Dict[str, Any], output_path: Path, llm_label: str
) -> None:
    """Write a human-readable markdown summary."""
    lines = [
        "# Evidence Retrieval Benchmark Results",
        "",
        "All ablations run on the bundled `SyntheticEHRNotesDataset` "
        "(5 patients, 8 note-condition samples, 5 positives, 3 "
        "negatives). Each row reports binary metrics on the Pass-1 "
        "note-level decision.",
        "",
        f"Backends compared: **offline stub** vs **{llm_label}**.",
        "",
    ]

    # Ablation 1
    lines += [
        "## Ablation I — sequential vs single-prompt",
        "",
        "| Backend | Prompt style | Accuracy | Precision | Recall | FP"
        " | Explanations |",
        "|---|---|:-:|:-:|:-:|:-:|:-:|",
    ]
    for backend_name in ("stub", "llm"):
        label = "offline stub" if backend_name == "stub" else llm_label
        for style in ("sequential", "single"):
            m = results["prompt_style"][backend_name][style]
            lines.append(
                f"| {label} | {style} | {m['accuracy']:.3f} | "
                f"{m['precision']:.3f} | {m['recall']:.3f} | {m['fp']} | "
                f"{m['generated_count']}/{m['total']} |"
            )
    lines.append("")

    # Ablation 2
    lines += [
        "## Ablation II — LLM vs CBERT-lite IR baseline",
        "",
        "| Model | Accuracy | Precision | Recall | FP |",
        "|---|:-:|:-:|:-:|:-:|",
    ]
    for backend_name in ("stub", "llm"):
        label = (
            "offline stub (sequential)"
            if backend_name == "stub"
            else f"{llm_label} (sequential)"
        )
        m = results["llm_vs_ir"][backend_name]["llm_sequential"]
        lines.append(
            f"| {label} | {m['accuracy']:.3f} | {m['precision']:.3f} | "
            f"{m['recall']:.3f} | {m['fp']} |"
        )
    m = results["llm_vs_ir"]["stub"]["cbert_lite"]
    lines.append(
        f"| CBERT-lite IR baseline | {m['accuracy']:.3f} | "
        f"{m['precision']:.3f} | {m['recall']:.3f} | {m['fp']} |"
    )
    lines.append("")

    # Ablation 3
    lines += [
        "## Ablation III — note-length budget sweep (novel axis)",
        "",
        "| Backend | max_note_chars | Accuracy | Precision | Recall | FP |",
        "|---|:-:|:-:|:-:|:-:|:-:|",
    ]
    for backend_name in ("stub", "llm"):
        label = "offline stub" if backend_name == "stub" else llm_label
        for budget, m in results["budget_sweep"][backend_name].items():
            lines.append(
                f"| {label} | {budget} | {m['accuracy']:.3f} | "
                f"{m['precision']:.3f} | {m['recall']:.3f} | {m['fp']} |"
            )
    lines.append("")

    output_path.write_text("\n".join(lines))


def _plot_charts(
    results: Dict[str, Any], output_dir: Path, llm_label: str
) -> None:
    """Render bar/line charts as PNGs. Uses matplotlib.pyplot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Chart 1: prompt-style ablation
    fig, ax = plt.subplots(figsize=(9, 5))
    metric_keys = ("accuracy", "precision", "recall")
    x_labels = ["sequential", "single"]
    x = range(len(x_labels))
    width = 0.13
    for idx, (backend_key, backend_label) in enumerate(
        (("stub", "stub"), ("llm", llm_label))
    ):
        for m_idx, metric in enumerate(metric_keys):
            values = [
                results["prompt_style"][backend_key][style][metric]
                for style in x_labels
            ]
            offset = (idx * 3 + m_idx - 2.5) * width
            ax.bar(
                [xi + offset for xi in x],
                values,
                width,
                label=f"{backend_label} {metric}",
            )
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Ablation I — sequential vs single-prompt")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "ablation1_prompt_style.png", dpi=150)
    plt.close(fig)

    # Chart 2: LLM vs CBERT-lite
    fig, ax = plt.subplots(figsize=(9, 5))
    categories = [
        f"{llm_label}\n(sequential)",
        "stub\n(sequential)",
        "CBERT-lite\nIR baseline",
    ]
    entries = [
        results["llm_vs_ir"]["llm"]["llm_sequential"],
        results["llm_vs_ir"]["stub"]["llm_sequential"],
        results["llm_vs_ir"]["stub"]["cbert_lite"],
    ]
    x = range(len(categories))
    width = 0.25
    for m_idx, metric in enumerate(metric_keys):
        values = [e[metric] for e in entries]
        ax.bar(
            [xi + (m_idx - 1) * width for xi in x],
            values,
            width,
            label=metric,
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Ablation II — LLM vs CBERT-lite IR baseline")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "ablation2_llm_vs_ir.png", dpi=150)
    plt.close(fig)

    # Chart 3: note-length budget sweep (line chart)
    fig, ax = plt.subplots(figsize=(9, 5))
    for backend_key, backend_label in (("stub", "stub"), ("llm", llm_label)):
        budgets = sorted(
            int(k) for k in results["budget_sweep"][backend_key].keys()
        )
        for metric in metric_keys:
            values = [
                results["budget_sweep"][backend_key][str(b)][metric]
                for b in budgets
            ]
            ax.plot(
                budgets,
                values,
                marker="o",
                label=f"{backend_label} {metric}",
            )
    ax.set_xscale("log")
    ax.set_xlabel("max_note_chars (log scale)")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Ablation III — note-length budget sweep (novel)")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "ablation3_budget_sweep.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run every ablation on stub + real-LLM backends."
    )
    parser.add_argument(
        "--llm-backend",
        choices=("stub", "openai"),
        default="openai",
        help="Real-LLM backend for the 'llm' column (default: openai).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model identifier for --llm-backend=openai.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR),
        help="Directory to write JSON, markdown, and PNG outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = _load_data()
    stub_factory = _build_backend_factory("stub", args.model)
    llm_factory = _build_backend_factory(args.llm_backend, args.model)
    llm_label = (
        args.model if args.llm_backend == "openai" else args.llm_backend
    )

    print(f"Running prompt-style ablation on stub…")
    ps_stub = _run_prompt_style_ablation(data, stub_factory)
    print(f"Running prompt-style ablation on {llm_label}…")
    ps_llm = _run_prompt_style_ablation(data, llm_factory)

    print(f"Running LLM-vs-IR comparison on stub…")
    ir_stub = _run_llm_vs_ir(data, stub_factory)
    print(f"Running LLM-vs-IR comparison on {llm_label}…")
    ir_llm = _run_llm_vs_ir(data, llm_factory)

    print(f"Running budget sweep on stub…")
    bs_stub = _run_note_budget_sweep(data, stub_factory)
    print(f"Running budget sweep on {llm_label}…")
    bs_llm = _run_note_budget_sweep(data, llm_factory)

    results = {
        "llm_label": llm_label,
        "llm_backend": args.llm_backend,
        "prompt_style": {"stub": ps_stub, "llm": ps_llm},
        "llm_vs_ir": {"stub": ir_stub, "llm": ir_llm},
        "budget_sweep": {"stub": bs_stub, "llm": bs_llm},
    }

    (output_dir / "benchmark_results.json").write_text(
        json.dumps(results, indent=2)
    )
    _write_markdown_report(
        results, output_dir / "benchmark_results.md", llm_label
    )
    _plot_charts(results, output_dir, llm_label)

    print()
    print(f"Wrote {output_dir / 'benchmark_results.json'}")
    print(f"Wrote {output_dir / 'benchmark_results.md'}")
    print(f"Wrote {output_dir / 'ablation1_prompt_style.png'}")
    print(f"Wrote {output_dir / 'ablation2_llm_vs_ir.png'}")
    print(f"Wrote {output_dir / 'ablation3_budget_sweep.png'}")


if __name__ == "__main__":
    main()
