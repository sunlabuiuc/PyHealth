"""PLM-CA for ICD-9 coding on MIMIC-III, with an input-length ablation.

Contributors: <Full Name> (<gt_username>), <Full Name> (<gt_username>)
Paper: Edin et al., "An Unsupervised Approach to Achieve Supervised-Level
    Explainability in Healthcare Records", EMNLP 2024.
Paper link: https://arxiv.org/abs/2406.08958

What this script does
---------------------
1. Builds the medical coding data. With ``--mimic3_root`` it follows the
   paper: MIMIC-III discharge summaries labelled with the admission's ICD-9
   diagnosis and procedure codes. With ``--synthetic`` it generates small
   fake notes in which each code is signalled by a few keywords, so the whole
   pipeline runs without credentials or downloads.
2. Keeps the ``--top_k`` most frequent codes (default 50), splits by patient
   and builds dataloaders.
3. Trains :class:`pyhealth.models.PLMCA` with the paper's optimizer settings
   (AdamW, learning rate 5e-5, dropout 0.2) via :class:`pyhealth.trainer.Trainer`.
4. With ``--ablation``, repeats training for several configurations and prints
   one results table.

Ablation: what is varied and why
--------------------------------
The paper truncates every note to 6000 tokens and states this has a negligible
cost because few notes are longer. ``max_length`` tests that claim: if the
statement holds, cutting notes to 2048 tokens should cost little, while 512
tokens should hurt because codes documented late in the note are lost.
``chunk_size`` is the window each encoder pass sees (PLM-ICD style chunked
encoding); larger chunks give tokens more context but use more memory.
The explanation-method ablation (Attention vs InputXGrad vs AttInGrad) is
added in a later step, once AttInGrad is merged.

How to run
----------
Synthetic data, tiny local encoder, CPU, about a minute::

    python examples/mimic3_icd9_coding_plm_ca.py --synthetic --ablation

MIMIC-III (needs PhysioNet access and a GPU)::

    python examples/mimic3_icd9_coding_plm_ca.py \\
        --mimic3_root /path/to/mimic-iii/1.4 \\
        --model_name <hugging-face-encoder> --top_k 50 --epochs 20 --ablation

On a single free GPU, train on a random subset and run one configuration per
session. Each run saves ``<output_dir>/<config>/test_scores.json``, and
``--summarize`` collects the saved runs into one table::

    python examples/mimic3_icd9_coding_plm_ca.py --mimic3_root ... \\
        --model_name ... --max_samples 4000 --epochs 3 --configs truncate_512
    python examples/mimic3_icd9_coding_plm_ca.py --summarize

If a session ends after training but before testing, ``--eval_only`` scores
the saved checkpoints without retraining (same data arguments as the run)::

    python examples/mimic3_icd9_coding_plm_ca.py --mimic3_root ... \\
        --model_name ... --max_samples 4000 --epochs 3 --configs chunk_256 \\
        --eval_only

Results
-------
Fill in from the printed table after running.

* Synthetic run: TODO
* MIMIC-III top-50: TODO

Deviations from the paper
-------------------------
* Random patient-level 80/10/10 split instead of the MIMIC-III full split of
  Mullenbach et al. (2018).
* Top-K codes instead of all ~8.9k codes, to keep the class-by-token
  attention matrix within a single GPU's memory.
* Fixed decision threshold of 0.3 (PyHealth's multilabel default) instead of
  tuning it on the validation set.
* No learning-rate warmup or linear decay, since ``Trainer`` has no scheduler.
* With ``--max_samples``, a seeded random subset of notes instead of the full
  training set, and fewer epochs, to fit a single free GPU.
"""

import argparse
import json
import os
import random
import tempfile
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from tqdm import tqdm
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from pyhealth.data import Patient
from pyhealth.datasets import (
    MIMIC3Dataset,
    SampleDataset,
    create_sample_dataset,
    get_dataloader,
    split_by_patient,
)
from pyhealth.metrics import multilabel_metrics_fn
from pyhealth.models import PLMCA
from pyhealth.tasks import MIMIC3ICD9Coding
from pyhealth.trainer import Trainer

TEXT_KEY = "text"
LABEL_KEY = "icd_codes"
METRICS = ["f1_micro", "f1_macro", "pr_auc_micro", "pr_auc_macro"]
# How often --eval_only saves its progress, so a disconnect loses little.
SAVE_EVERY_SECONDS = 600

# Configurations compared in the ablation. "base" comes first and is the only
# one run without --ablation.
SYNTHETIC_CONFIGS = [
    {"name": "base", "max_length": 128, "chunk_size": 32},
    {"name": "truncate_32", "max_length": 32, "chunk_size": 32},
    {"name": "chunk_16", "max_length": 128, "chunk_size": 16},
]
MIMIC_CONFIGS = [
    {"name": "base_6000", "max_length": 6000, "chunk_size": 128},
    {"name": "truncate_2048", "max_length": 2048, "chunk_size": 128},
    {"name": "truncate_512", "max_length": 512, "chunk_size": 128},
    {"name": "chunk_256", "max_length": 6000, "chunk_size": 256},
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class DischargeSummaryICD9Coding(MIMIC3ICD9Coding):
    """MIMIC3ICD9Coding restricted to discharge summaries, as in the paper.

    The parent task concatenates every note of an admission. The paper uses
    discharge summaries only, so this subclass filters notes by category.
    """

    task_name: str = "mimic3_icd9_coding_discharge"

    def __call__(self, patient: Patient) -> list[dict]:
        samples = []
        for admission in patient.get_events(event_type="admissions"):
            notes = patient.get_events(
                event_type="noteevents",
                filters=[
                    ("hadm_id", "==", admission.hadm_id),
                    ("category", "==", "Discharge summary"),
                ],
            )
            codes = set()
            for table in ("diagnoses_icd", "procedures_icd"):
                events = patient.get_events(
                    event_type=table, filters=[("hadm_id", "==", admission.hadm_id)]
                )
                codes.update(e.icd9_code for e in events if e.icd9_code)
            text = " ".join(note.text for note in notes)
            if text and codes:
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        TEXT_KEY: text,
                        LABEL_KEY: sorted(codes),
                    }
                )
        return samples


# Each synthetic code is "documented" by a few keywords placed at random
# positions in the note, so truncating the note can remove the evidence.
SYNTHETIC_CODES = {
    "428.0": ["heart", "failure", "edema"],
    "038.9": ["sepsis", "bacteremia"],
    "584.9": ["acute", "kidney", "injury"],
    "486": ["pneumonia", "infiltrate"],
    "401.9": ["hypertension"],
    "250.00": ["diabetes", "insulin"],
}
FILLER = [
    "patient", "admitted", "with", "history", "of", "was", "given", "and",
    "stable", "discharged", "home", "the", "on", "day", "pain", "noted",
]  # fmt: skip


def make_synthetic_samples(num_patients: int, seed: int) -> list[dict]:
    """Generates fake notes whose codes are signalled by keywords."""
    rng = random.Random(seed)
    samples = []
    for i in range(num_patients):
        codes = rng.sample(sorted(SYNTHETIC_CODES), k=rng.randint(1, 3))
        words = [rng.choice(FILLER) for _ in range(rng.randint(10, 90))]
        for code in codes:
            for keyword in SYNTHETIC_CODES[code]:
                words.insert(rng.randrange(len(words) + 1), keyword)
        samples.append(
            {"patient_id": f"p{i}", TEXT_KEY: " ".join(words), LABEL_KEY: codes}
        )
    return samples


def save_tiny_tokenizer(vocab: list[str], folder: Path) -> None:
    """Saves a BERT-style word-piece tokenizer for ``vocab`` to ``folder``.

    Built with the ``tokenizers`` library because ``BertTokenizerFast``
    silently ignores ``vocab_file`` in transformers 5.x, which would map
    every word to ``[UNK]``.
    """
    ids = {token: index for index, token in enumerate(vocab)}
    backend = Tokenizer(models.WordPiece(vocab=ids, unk_token="[UNK]"))
    backend.normalizer = normalizers.BertNormalizer(lowercase=True)
    backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", ids["[CLS]"]), ("[SEP]", ids["[SEP]"])],
    )
    PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    ).save_pretrained(folder)


def build_tiny_encoder(folder: Path) -> str:
    """Saves a small randomly initialised BERT so the demo needs no download."""
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *FILLER]
    vocab += sorted({w for words in SYNTHETIC_CODES.values() for w in words})
    save_tiny_tokenizer(vocab, folder)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
    )
    BertModel(config).save_pretrained(folder)
    return str(folder)


def keep_top_k_codes(
    dataset: SampleDataset,
    top_k: int,
    max_samples: int | None = None,
    seed: int = 42,
) -> SampleDataset:
    """Rebuilds the dataset with only the ``top_k`` most frequent codes.

    If ``max_samples`` is set, a seeded random subset of that many samples is
    kept first. Samples left with no code are dropped. The label processor is
    refitted, so the model's output layer has exactly ``top_k`` units.
    """
    label_vocab = dataset.output_processors[LABEL_KEY].label_vocab
    index_to_code = {index: code for code, index in label_vocab.items()}
    rows = [(s["patient_id"], s[TEXT_KEY], s[LABEL_KEY]) for s in dataset]
    if max_samples is not None and max_samples < len(rows):
        rows = random.Random(seed).sample(rows, max_samples)
    counts = torch.stack([target for _, _, target in rows]).sum(dim=0)
    kept = set(counts.topk(min(top_k, len(counts))).indices.tolist())

    samples = []
    for patient_id, text, target in rows:
        codes = [index_to_code[i] for i in target.nonzero().flatten().tolist()]
        codes = [c for c in codes if label_vocab[c] in kept]
        if codes:
            samples.append({"patient_id": patient_id, TEXT_KEY: text, LABEL_KEY: codes})
    return create_sample_dataset(
        samples=samples,
        input_schema={TEXT_KEY: "text"},
        output_schema={LABEL_KEY: "multilabel"},
        dataset_name="mimic3_top_k",
        task_name=f"icd9_coding_top{top_k}",
    )


def load_data(args: argparse.Namespace) -> SampleDataset:
    """Returns the coding dataset, from MIMIC-III or synthetic notes."""
    if args.synthetic:
        samples = make_synthetic_samples(args.num_synthetic, args.seed)
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={TEXT_KEY: "text"},
            output_schema={LABEL_KEY: "multilabel"},
            dataset_name="synthetic_notes",
        )
    else:
        base = MIMIC3Dataset(
            root=args.mimic3_root,
            tables=["diagnoses_icd", "procedures_icd", "noteevents"],
            dev=args.dev,
        )
        dataset = base.set_task(DischargeSummaryICD9Coding())
    return keep_top_k_codes(dataset, args.top_k, args.max_samples, args.seed)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def run_config(
    config: dict,
    dataset: SampleDataset,
    splits: tuple,
    model_name: str,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Trains PLM-CA with one configuration and returns its test scores."""
    train_ds, val_ds, test_ds = splits
    torch.manual_seed(args.seed)  # same initial decoder for every config
    model = PLMCA(
        dataset,  # full dataset: its fitted processors define the label space
        model_name=model_name,
        chunk_size=config["chunk_size"],
        max_length=config["max_length"],
        dropout=0.2,
    )
    trainer = Trainer(
        model=model,
        metrics=METRICS,
        output_path=args.output_dir,
        exp_name=config["name"],
    )
    train_loader = get_dataloader(
        train_ds, args.batch_size, shuffle=True, seed=args.seed
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=get_dataloader(val_ds, args.batch_size),
        epochs=args.epochs,
        optimizer_class=torch.optim.AdamW,
        optimizer_params={"lr": args.lr},
        monitor="f1_micro",
        monitor_criterion="max",
    )
    scores = trainer.evaluate(get_dataloader(test_ds, args.batch_size))
    save_scores(config, dataset, scores, args)
    return scores


def evaluate_saved_run(
    config: dict,
    dataset: SampleDataset,
    splits: tuple,
    model_name: str,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Scores a run whose training finished but whose testing did not.

    Picks between the run's ``best.ckpt`` and ``last.ckpt`` by validation
    micro-F1, as ``Trainer`` does at the end of training, then evaluates the
    chosen checkpoint on the test split and saves ``test_scores.json``.

    Every scoring step saves its progress under ``<run>/eval_progress/``, so
    a rerun after a disconnect continues where the last one stopped.
    """
    _, val_ds, test_ds = splits
    run_dir = Path(args.output_dir) / config["name"]
    progress_dir = run_dir / "eval_progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = PLMCA(
        dataset,
        model_name=model_name,
        chunk_size=config["chunk_size"],
        max_length=config["max_length"],
        dropout=0.2,
    ).to(device)

    def score(checkpoint: str, split: str, split_ds) -> dict[str, float]:
        progress_path = progress_dir / f"{Path(checkpoint).stem}_{split}.pt"
        if not progress_path.with_suffix(".json").is_file():
            state = torch.load(
                run_dir / checkpoint, map_location=device, weights_only=True
            )
            model.load_state_dict(state)
        loader = get_dataloader(split_ds, args.batch_size)
        return evaluate_resumable(model, loader, progress_path)

    checkpoints = [n for n in ("best.ckpt", "last.ckpt") if (run_dir / n).is_file()]
    if not checkpoints:
        raise FileNotFoundError(f"No best.ckpt or last.ckpt in {run_dir}")
    chosen, chosen_score = None, float("-inf")
    for name in checkpoints:
        value = score(name, "val", val_ds)["f1_micro"]
        print(f"{config['name']}/{name}: validation f1_micro {value:.4f}")
        if value > chosen_score:  # strictly better, as in Trainer
            chosen, chosen_score = name, value
    scores = score(chosen, "test", test_ds)
    save_scores(config, dataset, scores, args, checkpoint=chosen)
    return scores


def evaluate_resumable(
    model: PLMCA, dataloader, progress_path: Path
) -> dict[str, float]:
    """Evaluates like ``Trainer.evaluate``, saving progress to resume later.

    Predictions so far are saved to ``progress_path`` every
    ``SAVE_EVERY_SECONDS``. A rerun skips the batches already scored. When
    all batches are done, the scores are written next to it as ``.json`` and
    a later rerun returns them without evaluating again.
    """
    done_path = progress_path.with_suffix(".json")
    if done_path.is_file():
        return json.loads(done_path.read_text())
    state = {"batches": 0, "y_true": [], "y_prob": [], "loss": []}
    if progress_path.is_file():
        state = torch.load(progress_path, weights_only=True)
        print(f"Resuming {progress_path.stem} after batch {state['batches']}")

    model.eval()
    last_save = time.monotonic()
    with torch.no_grad():
        for index, batch in enumerate(tqdm(dataloader, desc=progress_path.stem)):
            if index < state["batches"]:
                continue  # scored in an earlier session
            output = model(**batch)
            state["y_true"].append(output["y_true"].cpu())
            state["y_prob"].append(output["y_prob"].cpu())
            state["loss"].append(output["loss"].item())
            state["batches"] = index + 1
            if time.monotonic() - last_save >= SAVE_EVERY_SECONDS:
                save_atomically(state, progress_path)
                last_save = time.monotonic()

    scores = multilabel_metrics_fn(
        torch.cat(state["y_true"]).numpy(),
        torch.cat(state["y_prob"]).numpy(),
        metrics=METRICS,
    )
    scores = {key: float(value) for key, value in scores.items()}
    scores["loss"] = sum(state["loss"]) / len(state["loss"])
    done_path.write_text(json.dumps(scores, indent=2))
    progress_path.unlink(missing_ok=True)
    return scores


def save_atomically(state: dict, path: Path) -> None:
    """Saves to a temporary file first, so a disconnect cannot corrupt it."""
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def save_scores(
    config: dict,
    dataset: SampleDataset,
    scores: dict[str, float],
    args: argparse.Namespace,
    checkpoint: str | None = None,
) -> None:
    """Writes ``<output_dir>/<config>/test_scores.json`` for ``--summarize``."""
    record = {
        **config,
        "num_samples": len(dataset),
        "epochs": args.epochs,
        **{key: float(value) for key, value in scores.items()},
    }
    if checkpoint is not None:
        record["scored_from_checkpoint"] = checkpoint
    out_dir = Path(args.output_dir) / config["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_scores.json").write_text(json.dumps(record, indent=2))


def print_table(results: dict[str, dict[str, float]]) -> None:
    """Prints one row per configuration with the test metrics."""
    header = f"{'config':<16}" + "".join(f"{m:>14}" for m in METRICS)
    print("\n" + header + "\n" + "-" * len(header))
    for name, scores in results.items():
        print(f"{name:<16}" + "".join(f"{scores[m]:>14.4f}" for m in METRICS))


def load_saved_results(output_dir: str) -> dict[str, dict[str, float]]:
    """Reads every ``<output_dir>/<config>/test_scores.json`` saved so far."""
    results = {}
    for path in sorted(Path(output_dir).glob("*/test_scores.json")):
        record = json.loads(path.read_text())
        results[record["name"]] = record
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    data = parser.add_mutually_exclusive_group()
    data.add_argument("--synthetic", action="store_true")
    data.add_argument("--mimic3_root", type=str)
    data.add_argument(
        "--summarize",
        action="store_true",
        help="print a table of the runs already saved in --output_dir",
    )
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="names of the configurations to run (default: base, or all "
        "with --ablation)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="train on a seeded random subset of this many notes",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="score the checkpoints saved in --output_dir instead of training",
    )
    parser.add_argument("--dev", action="store_true", help="MIMIC-III dev subset")
    parser.add_argument("--num_synthetic", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="output/plm_ca")
    args = parser.parse_args()

    # Synthetic defaults: a randomly initialised tiny encoder needs a larger
    # learning rate and a few more epochs than fine-tuning a real one.
    if args.epochs is None:
        args.epochs = 15 if args.synthetic else 20
    if args.batch_size is None:
        args.batch_size = 8
    if args.lr is None:
        args.lr = 1e-3 if args.synthetic else 5e-5
    if not (args.synthetic or args.mimic3_root or args.summarize):
        parser.error("one of --synthetic, --mimic3_root or --summarize is required")
    if args.mimic3_root and args.model_name is None:
        parser.error("--model_name is required with --mimic3_root")
    if args.configs:
        known = SYNTHETIC_CONFIGS if args.synthetic else MIMIC_CONFIGS
        unknown = set(args.configs) - {config["name"] for config in known}
        if unknown:
            parser.error(
                f"unknown config(s) {sorted(unknown)}; choose from "
                f"{[config['name'] for config in known]}"
            )
    return args


def main() -> None:
    args = parse_args()
    if args.summarize:
        results = load_saved_results(args.output_dir)
        if not results:
            print(f"No saved runs found in {args.output_dir}")
            return
        print_table(results)
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = load_data(args)
    splits = split_by_patient(dataset, [0.8, 0.1, 0.1], seed=args.seed)
    num_codes = dataset.output_processors[LABEL_KEY].size()
    print(f"samples: {len(dataset)}, codes: {num_codes}")
    print("train/val/test sizes:", [len(split) for split in splits])

    configs = SYNTHETIC_CONFIGS if args.synthetic else MIMIC_CONFIGS
    if args.configs:
        configs = [config for config in configs if config["name"] in args.configs]
    elif not args.ablation:
        configs = configs[:1]

    with tempfile.TemporaryDirectory() as tmp:
        model_name = args.model_name or build_tiny_encoder(Path(tmp))
        run = evaluate_saved_run if args.eval_only else run_config
        results = {
            config["name"]: run(config, dataset, splits, model_name, args)
            for config in configs
        }
    print_table(results)


if __name__ == "__main__":
    # The guard is required: PyHealth uses multiprocessing internally.
    main()
