"""EviCare next-visit diagnosis prediction on MIMIC-III or MIMIC-IV, end to end.

Contributor: Kavish Shah (GitHub: thekavishshah)

Paper: Hengyu Zhang et al. EviCare: Enhancing Diagnosis Prediction with Deep
Model-Guided Evidence for In-Context Reasoning. KDD 2026.
https://arxiv.org/abs/2604.10455, https://doi.org/10.1145/3770854.3780257

Pipeline:
    1. Load diagnoses, map ICD codes to CCS and build one sample per visit
       after the first (``EviCareDiagnosisPredictionMIMIC3`` / ``MIMIC4``),
       in the ``overall`` or ``novel`` setting.
    2. Split the eligible patients (at least two admissions with mapped
       diagnoses) 7:1:2 and optionally keep a fraction of the training
       patients (the paper's main tables train the guide on 5%). The split
       is over patients, not samples, so patients that give no sample (in
       the novel setting) still belong to a split.
    3. Train the guide model (RETAIN, embedding size 16) and keep the epoch
       with the lowest validation loss.
    4. Fit the co-occurrence graph on the training patients only, from all
       CCS codes of all of their admissions (``task.patient_code_sets``),
       whatever the setting.
    5. Evaluate the guide alone, then EviCare with an LLM, on the test split:
       visit-level P@k and code-level Acc@k (k = 5, 10 for novel; 10, 20 for
       overall, as in the paper).

Usage:
    # Offline smoke run on the MIMIC-III demo shipped with PyHealth, with a
    # fake LLM (a deterministic heuristic, not a model). About a minute on a
    # laptop CPU.
    python examples/diagnosis_prediction/evicare_mimic3.py --fake-llm

    # Same on the MIMIC-IV demo shipped with PyHealth, novel setting.
    python examples/diagnosis_prediction/evicare_mimic3.py \\
        --dataset mimic4 --setting novel --fake-llm

    # A real LLM behind an OpenAI-compatible endpoint, e.g. vLLM serving the
    # paper's model on your own machine:
    #   vllm serve Qwen/Qwen3-8B --port 8000
    python examples/diagnosis_prediction/evicare_mimic3.py \\
        --llm Qwen/Qwen3-8B --base-url http://localhost:8000/v1

    # Endpoints that reject top_k or chat_template_kwargs (vLLM-specific
    # fields; e.g. api.openai.com returns HTTP 400): send only the standard
    # temperature / top_p / max_tokens.
    python examples/diagnosis_prediction/evicare_mimic3.py \\
        --llm <model> --base-url <url> --no-extra-body

    # Paper setting (credentialed MIMIC-III, a GPU, a local LLM server).
    python examples/diagnosis_prediction/evicare_mimic3.py \\
        --root /path/to/mimic-iii/1.4 --setting novel --train-fraction 0.05 \\
        --epochs 50 --llm Qwen/Qwen3-8B --base-url http://localhost:8000/v1

Data use: the PhysioNet credentialed data use agreement does not allow
sending MIMIC records to third-party online services. With ``--root``
pointing at credentialed data, this script only calls LLM servers on this
machine (``localhost``) unless you pass ``--allow-remote-llm`` for data you
are allowed to share (for example the open demo subsets).

The demo datasets have about a hundred patients, so the test split holds a
handful of visits and the printed metrics are noise. The demo runs only show
that the pipeline works; they are not a reproduction of the paper.
"""

import argparse
import json
import random
import re
import urllib.parse
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import MIMIC3Dataset, MIMIC4EHRDataset, get_dataloader
from pyhealth.models import EviCare
from pyhealth.tasks import (
    EviCareDiagnosisPredictionMIMIC3,
    EviCareDiagnosisPredictionMIMIC4,
)

RESOURCES = Path(__file__).resolve().parents[2] / "test-resources" / "core"
DEMO_ROOTS = {"mimic3": RESOURCES / "mimic3demo", "mimic4": RESOURCES / "mimic4demo"}

# Decoding defaults: the settings of the official repository's inference
# config for Qwen3-8B (model/qwen3-8B.yaml in github.com/zhyccc/EviCare),
# i.e. Qwen3's non-thinking mode with a short reply budget.
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
DEFAULT_MAX_NEW_TOKENS = 256

DEMO_NOTE = (
    "NOTE: the demo split is tiny (a handful of test visits). The metrics "
    "below only show that the pipeline runs; they are not results and are "
    "not comparable with the paper."
)


# ----------------------------------------------------------------------------
# LLMs
# ----------------------------------------------------------------------------
class FakeLLM:
    """An offline stand-in for an LLM, for smoke runs. It is not a model.

    It reads the candidate list and the relational evidence from the prompt,
    puts the candidates that have relational support first and the others
    after, both in prompt order. To exercise the answer parser it also adds
    an unknown category and repeats one name.
    """

    def __call__(self, system: str, prompt: str) -> str:
        lines = prompt.splitlines()
        header = next(
            i for i, line in enumerate(lines) if line.startswith("Candidate CCS")
        )
        candidates = re.findall(r'"([^"]*)"', lines[header + 1])
        supported = {
            match.group(1)
            for line in lines
            if (match := re.search(r' => "([^"]*)"$', line))
        }
        ranked = [c for c in candidates if c in supported]
        ranked += [c for c in candidates if c not in supported]
        noisy = ["Not a real category"] + ranked[:1] + ranked
        return "Thinking it over.\nAnswer: " + ", ".join(noisy)


def is_local(url: str | None) -> bool:
    """Whether an endpoint URL points at this machine."""
    if not url:
        return False
    host = urllib.parse.urlparse(url).hostname or ""
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")


def make_llm(args, root_is_demo: bool):
    """Builds the LLM callable chosen on the command line (or None)."""
    if args.fake_llm:
        return FakeLLM()
    if not args.llm:
        return None
    if not (root_is_demo or is_local(args.base_url) or args.allow_remote_llm):
        raise SystemExit(
            "Refusing to send records from --root to a remote LLM: the "
            "PhysioNet data use agreement forbids sharing credentialed MIMIC "
            "data with online services. Serve the model on this machine "
            "(--base-url http://localhost:8000/v1) or, for data you may "
            "share, pass --allow-remote-llm."
        )
    from pyhealth.autoimprove import OpenAILLM

    # vLLM and SGLang accept top_k and Qwen3's chat-template switch in the
    # request body; other OpenAI-compatible endpoints may reject them, so
    # --no-extra-body sends only the standard sampling fields.
    extra_body = {"temperature": args.temperature, "top_p": args.top_p}
    if not args.no_extra_body:
        extra_body["top_k"] = args.top_k
        extra_body["chat_template_kwargs"] = {
            "enable_thinking": args.enable_thinking
        }
    return OpenAILLM(
        model=args.llm,
        base_url=args.base_url,
        max_tokens=args.max_new_tokens,
        extra_body=extra_body,
    )


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_samples(args):
    """Loads the dataset, applies the EviCare task and reads patient codes.

    Returns the task, the samples and, for every eligible patient, all of
    their CCS codes (the rows of the co-occurrence matrix ``A``).
    """
    root = args.root or str(DEMO_ROOTS[args.dataset])
    if args.dataset == "mimic3":
        dataset = MIMIC3Dataset(root=root, tables=["diagnoses_icd"])
        task_class = EviCareDiagnosisPredictionMIMIC3
    else:
        dataset = MIMIC4EHRDataset(root=root, tables=["diagnoses_icd"])
        task_class = EviCareDiagnosisPredictionMIMIC4
    task = task_class(setting=args.setting, ccs_mapping=args.ccs_mapping)
    samples = dataset.set_task(task, num_workers=args.num_workers)
    return task, samples, task.patient_code_sets(dataset)


def split_patients(samples, patient_ids, train_fraction: float, seed: int):
    """Splits patients 7:1:2 and keeps a fraction of the training patients.

    Returns the train, validation and test sample subsets and the kept
    training patient IDs (including those without samples).
    """
    rng = random.Random(seed)
    patients = sorted({str(p) for p in patient_ids} | set(samples.patient_to_index))
    rng.shuffle(patients)
    n_train, n_val = int(len(patients) * 0.7), int(len(patients) * 0.1)
    parts = [
        patients[:n_train],
        patients[n_train : n_train + n_val],
        patients[n_train + n_val :],
    ]
    if train_fraction < 1.0:
        keep = max(1, round(len(parts[0]) * train_fraction))
        parts[0] = parts[0][:keep]
    subsets = [
        samples.subset(
            [i for p in part for i in samples.patient_to_index.get(p, [])]
        )
        for part in parts
    ]
    return (*subsets, parts[0])


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def summarize(name: str, scores: dict[str, float]) -> None:
    text = "  ".join(f"{key} {100 * value:6.2f}" for key, value in scores.items())
    print(f"{name:<24}{text}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=("mimic3", "mimic4"), default="mimic3")
    parser.add_argument(
        "--root", default=None, help="Dataset root. Default: the bundled demo."
    )
    parser.add_argument("--setting", choices=("overall", "novel"), default="overall")
    parser.add_argument(
        "--ccs-mapping",
        default=None,
        help="CSV with icd_code, ccs_code[, icd_version, ccs_name, icd_name] "
        "(e.g. CCSR). Default: PyHealth's single-level CCS tables.",
    )
    parser.add_argument("--k", type=int, default=50, help="Number of candidates.")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Fraction of training patients used to train the guide and fit "
        "the co-occurrence graph (the paper's main tables use 0.05).",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--max-prompt-visits",
        type=int,
        default=None,
        help="Overall setting: show only the most recent visits in the prompt.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Evaluate on at most this many test samples (limits LLM calls).",
    )
    llm = parser.add_argument_group("LLM")
    llm.add_argument("--llm", default=None, help="Model name on the endpoint.")
    llm.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1. "
        "Default: OPENAI_BASE_URL, else api.openai.com.",
    )
    llm.add_argument(
        "--fake-llm", action="store_true", help="Offline heuristic, no model."
    )
    llm.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    llm.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    llm.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    llm.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    llm.add_argument(
        "--no-extra-body",
        action="store_true",
        help="Do not send top_k and chat_template_kwargs (vLLM/SGLang fields "
        "that other OpenAI-compatible endpoints may reject).",
    )
    llm.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Turn on Qwen3's thinking mode (off in the official config).",
    )
    llm.add_argument(
        "--allow-remote-llm",
        action="store_true",
        help="Allow a non-local endpoint for data passed with --root. Only for "
        "data you may share with that service.",
    )
    parser.add_argument(
        "--save-predictions",
        default=None,
        help="Write prompts, replies and rankings to this JSON file.",
    )
    args = parser.parse_args()
    if args.fake_llm and args.llm:
        parser.error("Use either --fake-llm or --llm, not both.")
    if not 0.0 < args.train_fraction <= 1.0:
        parser.error("--train-fraction must be in (0, 1].")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root_is_demo = args.root is None
    llm_fn = make_llm(args, root_is_demo)

    task, samples, code_sets = load_samples(args)
    train, val, test, train_patients = split_patients(
        samples, code_sets, args.train_fraction, args.seed
    )
    if args.max_test_samples is not None:
        test = test.subset(list(range(min(args.max_test_samples, len(test)))))
    print(
        f"{len(samples)} samples; train/val/test {len(train)}/{len(val)}/"
        f"{len(test)} ({args.setting} setting)"
    )
    if len(train) == 0 or len(test) == 0:
        raise SystemExit("Not enough patients for a train and a test split.")

    model = EviCare(
        samples,
        setting=args.setting,
        k=args.k,
        ccs_names=task.ccs_mapping.ccs_names,
        max_prompt_visits=args.max_prompt_visits,
        embedding_dim=args.embedding_dim,
    )
    from pyhealth.trainer import Trainer

    has_val = len(val) > 0
    Trainer(model=model, enable_logging=False).train(
        train_dataloader=get_dataloader(train, batch_size=args.batch_size, shuffle=True),
        val_dataloader=(
            get_dataloader(val, batch_size=args.batch_size) if has_val else None
        ),
        epochs=args.epochs,
        optimizer_params={"lr": args.lr},
        monitor="loss" if has_val else None,
        monitor_criterion="min",
    )
    model.fit_cooccurrence({p: code_sets[p] for p in train_patients if p in code_sets})

    ks = (5, 10) if args.setting == "novel" else (10, 20)
    print()
    if root_is_demo:
        print(DEMO_NOTE)
    summarize("Guide (RETAIN)", model.evaluate(model.predict(test), ks))
    if llm_fn is None:
        print("No LLM given (--llm or --fake-llm): only the guide was evaluated.")
        return

    predictions = model.predict(test, llm=llm_fn)
    label = "EviCare (fake LLM)" if args.fake_llm else f"EviCare ({args.llm})"
    summarize(label, model.evaluate(predictions, ks))
    failed = sum(p.error is not None for p in predictions)
    parsed = [p.n_parsed for p in predictions if p.error is None]
    print(
        f"LLM calls: {len(predictions)}, failed: {failed}; codes parsed per "
        f"reply: mean {np.mean(parsed) if parsed else 0:.1f} of "
        f"{np.mean([len(p.candidates) for p in predictions]):.1f} candidates"
    )
    if args.save_predictions:
        with open(args.save_predictions, "w", encoding="utf-8") as handle:
            json.dump([vars(p) for p in predictions], handle, indent=1, default=str)
        print(f"Saved predictions to {args.save_predictions}")


if __name__ == "__main__":
    main()
