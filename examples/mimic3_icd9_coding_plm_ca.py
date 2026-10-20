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
"""

import argparse
import random
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from pyhealth.data import Patient
from pyhealth.datasets import (
    MIMIC3Dataset,
    SampleDataset,
    create_sample_dataset,
    get_dataloader,
    split_by_patient,
)
from pyhealth.models import PLMCA
from pyhealth.tasks import MIMIC3ICD9Coding
from pyhealth.trainer import Trainer

TEXT_KEY = "text"
LABEL_KEY = "icd_codes"
METRICS = ["f1_micro", "f1_macro", "pr_auc_micro", "pr_auc_macro"]

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


def keep_top_k_codes(dataset: SampleDataset, top_k: int) -> SampleDataset:
    """Rebuilds the dataset with only the ``top_k`` most frequent codes.

    Samples left with no code are dropped. The label processor is refitted,
    so the model's output layer has exactly ``top_k`` units.
    """
    label_vocab = dataset.output_processors[LABEL_KEY].label_vocab
    index_to_code = {index: code for code, index in label_vocab.items()}
    rows = [(s["patient_id"], s[TEXT_KEY], s[LABEL_KEY]) for s in dataset]
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
    return keep_top_k_codes(dataset, args.top_k)


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
    return trainer.evaluate(get_dataloader(test_ds, args.batch_size))


def print_table(results: dict[str, dict[str, float]]) -> None:
    """Prints one row per configuration with the test metrics."""
    header = f"{'config':<16}" + "".join(f"{m:>14}" for m in METRICS)
    print("\n" + header + "\n" + "-" * len(header))
    for name, scores in results.items():
        print(f"{name:<16}" + "".join(f"{scores[m]:>14.4f}" for m in METRICS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--synthetic", action="store_true")
    data.add_argument("--mimic3_root", type=str)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--ablation", action="store_true")
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
    if args.mimic3_root and args.model_name is None:
        parser.error("--model_name is required with --mimic3_root")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = load_data(args)
    splits = split_by_patient(dataset, [0.8, 0.1, 0.1], seed=args.seed)
    num_codes = dataset.output_processors[LABEL_KEY].size()
    print(f"samples: {len(dataset)}, codes: {num_codes}")
    print("train/val/test sizes:", [len(split) for split in splits])

    configs = SYNTHETIC_CONFIGS if args.synthetic else MIMIC_CONFIGS
    if not args.ablation:
        configs = configs[:1]

    with tempfile.TemporaryDirectory() as tmp:
        model_name = args.model_name or build_tiny_encoder(Path(tmp))
        results = {
            config["name"]: run_config(config, dataset, splits, model_name, args)
            for config in configs
        }
    print_table(results)


if __name__ == "__main__":
    # The guard is required: PyHealth uses multiprocessing internally.
    main()
