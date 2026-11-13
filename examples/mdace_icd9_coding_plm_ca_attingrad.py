"""PLM-CA or CAML on MIMIC-III + MDACE: training and AttInGrad explanation ablation.

Contributors: Juri Moon (jmoon326)
Paper: Edin et al., "An Unsupervised Approach to Achieve Supervised-Level
    Explainability in Healthcare Records", EMNLP 2024.
Paper link: https://arxiv.org/abs/2406.08958

Stage 1, ``--train``: trains :class:`pyhealth.models.PLMCA` (default) or,
with ``--model caml``, :class:`pyhealth.models.CAML` (Mullenbach et al.,
NAACL 2018) on the
MIMIC-III discharge summaries of the paper's ``mdace_icd9_inpatient_code``
split (MIMIC-III full + MDACE train), using only the ICD-9 codes that occur
in MDACE so a laptop GPU can fit it.

Stage 2, ``--evaluate``: explains every (MDACE note, gold code) pair with
Attention, InputXGrad, AttInGrad, Integrated Gradients and attention x IG,
picks each method's threshold on MDACE val and reports token precision,
recall, F1, Empty and recall@5 on MDACE test.

No credentials? ``--synthetic`` runs both stages in about a minute on CPU:
fake notes where each code is signalled by a known keyword phrase (its
evidence span), plus filler and distractor words, and a tiny local encoder.

    python examples/mdace_icd9_coding_plm_ca_attingrad.py --synthetic

Real data: run the authors' preparation scripts first (github.com/JoakimEdin/
explainable-medical-coding: ``make mimiciii`` and ``make mdace``) and pass
their ``data/processed`` folder as ``--processed``. MIMIC-III needs
PhysioNet credentialed access; never commit the files.

    python examples/mdace_icd9_coding_plm_ca_attingrad.py --train \\
        --processed /path/to/data/processed \\
        --model_name /path/to/roberta-base-pm-m3-voc-hf --device mps
    python examples/mdace_icd9_coding_plm_ca_attingrad.py --evaluate \\
        --processed /path/to/data/processed \\
        --model_name /path/to/roberta-base-pm-m3-voc-hf --device mps \\
        --checkpoint output/mdace/plm_ca_mdace/best.ckpt

Add ``--model caml`` to train or evaluate CAML instead. It uses the same
tokenizer (so MDACE spans map to the same tokens), initialises its word
embeddings from ``--model_name`` and saves under ``caml_mdace/`` instead of
``plm_ca_mdace/``. Unless overridden, CAML uses the CAML paper's learning rate
1e-4 and batch size 16 with no accumulation.

Add ``--tm --init_checkpoint path/to/best.ckpt`` to fine-tune a trained model
with token masking (:func:`pyhealth.interpret.methods.token_masking_loss`):
one epoch at constant lr 1e-5, distilling from a frozen copy of the starting
model. It saves under ``{model}_tm_mdace/``.

Deviations from the paper (for compute): label space restricted to MDACE
codes, notes truncated to 4096 tokens instead of 6000, 2 epochs instead of
20, fixed 0.5 decision threshold. Training uses 10% linear warmup then
linear decay (``--warmup``) for both models.
"""

import argparse
import copy
import math
import os
import random
import tempfile
from pathlib import Path

import polars as pl
import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from tqdm import tqdm
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.interpret.methods import AttInGrad, token_masking_loss
from pyhealth.metrics.interpretability import (
    best_plausibility_threshold,
    plausibility_metrics,
    spans_to_token_ids,
)
from pyhealth.models import CAML, PLMCA
from pyhealth.trainer import Trainer

TEXT_KEY = "text"
LABEL_KEY = "icd_codes"
SPLITS = ("train", "val", "test")
EXPLANATION_METHODS = (
    "attention",
    "inputxgrad",
    "attingrad",
    "integrated_gradients",
    "att_ig",
)

# Synthetic demo: each fake code is signalled by one phrase (its evidence).
SYNTHETIC_CODES = {
    "428.0": "heart failure",
    "571.2": "alcoholic cirrhosis",
    "276.1": "low sodium",
    "038.9": "septic shock",
    "584.9": "kidney injury",
    "54.91": "paracentesis done",
}
# Clinical-sounding words that never decide a code, plus plain filler.
DISTRACTORS = ["edema", "fever", "dyspnea", "lasix", "crackles", "lactate"]
FILLER = (
    "patient was admitted with and the on day stable plan continue seen by "
    "team family at home no new pain review labs".split()
)


def write_synthetic_data(folder: Path, seed: int) -> None:
    """Writes fake notes in the same layout as the authors' processed data."""
    rng = random.Random(seed)
    sizes = {"train": 400, "val": 60, "test": 60}
    note_id = 0
    for split, size in sizes.items():
        rows = []
        for _ in range(size):
            codes = rng.sample(sorted(SYNTHETIC_CODES), k=rng.randint(1, 3))
            pieces = [rng.choice(FILLER + DISTRACTORS) for _ in range(40)]
            for code in codes:
                pieces.insert(rng.randrange(len(pieces) + 1), SYNTHETIC_CODES[code])
            text = " ".join(pieces) + " ."
            spans = []
            for code in codes:
                start = text.index(SYNTHETIC_CODES[code])
                spans.append([[start, start + len(SYNTHETIC_CODES[code])]])
            note_id += 1
            rows.append(
                {
                    "note_id": str(note_id),
                    "subject_id": note_id,
                    "_id": note_id,
                    "note_type": "Discharge summary",
                    "text": text,
                    "diagnosis_codes": codes,
                    "diagnosis_code_spans": spans,
                    "procedure_codes": [],
                    "procedure_code_spans": [],
                }
            )
        df = pl.DataFrame(rows)
        for name in ("mdace_icd9_inpatient", "mdace_icd9_inpatient_code"):
            (folder / name).mkdir(parents=True, exist_ok=True)
            df.write_parquet(folder / name / f"{split}.parquet")


def build_tiny_encoder(folder: Path) -> str:
    """Saves a small random BERT with a word-level vocabulary for the demo."""
    words = {w for p in SYNTHETIC_CODES.values() for w in p.split()}
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "."]
    vocab += sorted(words | set(FILLER) | set(DISTRACTORS))
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
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    BertModel(config).save_pretrained(folder)
    return str(folder)


def run_synthetic(args: argparse.Namespace) -> None:
    """Trains and evaluates on fake notes in a temporary folder."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_synthetic_data(root / "processed", args.seed)
        args.processed = str(root / "processed")
        args.model_name = build_tiny_encoder(root / "encoder")
        args.output_dir = str(root / "output")
        args.device = args.device or "cpu"
        args.max_length, args.chunk_size, args.batch_size = 128, 32, 16
        tm, tm_epochs, tm_lr = args.tm, args.epochs, args.lr
        args.tm = False
        args.epochs, args.lr, args.ig_steps, args.accum = 15, 1e-3, 8, 1
        train(args)
        if tm:
            # TM fine-tunes the base model just trained, as in the paper.
            args.init_checkpoint = str(root / "output" / exp_name(args) / "best.ckpt")
            args.tm, args.epochs, args.lr = True, tm_epochs, tm_lr
            train(args)
        args.checkpoint = str(root / "output" / exp_name(args) / "best.ckpt")
        evaluate(args)


# Per-model defaults for options the user did not set: (lr, batch_size, accum).
MODEL_DEFAULTS = {"plm_ca": (5e-5, 2, 8), "caml": (1e-4, 16, 1)}


def exp_name(args: argparse.Namespace) -> str:
    """Checkpoint folder name, separate per model so runs never overwrite."""
    return f"{args.model}_tm_mdace" if args.tm else f"{args.model}_mdace"


def build_model(args: argparse.Namespace, dataset) -> PLMCA | CAML:
    """Builds the model chosen with ``--model``."""
    if args.model == "caml":
        return CAML(dataset, model_name=args.model_name, max_length=args.max_length)
    return PLMCA(
        dataset,
        model_name=args.model_name,
        chunk_size=args.chunk_size,
        max_length=args.max_length,
        dropout=0.2,
    )


def note_codes(row: dict) -> list[str]:
    return list(row["diagnosis_codes"] or []) + list(row["procedure_codes"] or [])


def mdace_label_space(processed: Path) -> set[str]:
    """ICD-9 codes annotated anywhere in MDACE (train, val and test)."""
    codes = set()
    for split in SPLITS:
        df = pl.read_parquet(processed / "mdace_icd9_inpatient" / f"{split}.parquet")
        for row in df.iter_rows(named=True):
            codes.update(note_codes(row))
    return codes


def coding_samples(processed: Path, split: str, labels: set[str]) -> list[dict]:
    """Discharge summaries of one split, labelled with codes in ``labels``."""
    df = pl.read_parquet(processed / "mdace_icd9_inpatient_code" / f"{split}.parquet")
    return [
        {
            "patient_id": str(row["subject_id"]),
            "note_id": str(row["note_id"]),
            TEXT_KEY: row["text"],
            LABEL_KEY: sorted(set(note_codes(row)) & labels),
        }
        for row in df.iter_rows(named=True)
    ]


def training_samples(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    """Train and val samples exactly as used for training (same subset)."""
    processed = Path(args.processed)
    labels = mdace_label_space(processed)
    train_samples = coding_samples(processed, "train", labels)
    val_samples = coding_samples(processed, "val", labels)
    if args.max_train:
        random.Random(args.seed).shuffle(train_samples)
        train_samples = train_samples[: args.max_train]
    return train_samples, val_samples


def train(args: argparse.Namespace) -> None:
    processed = Path(args.processed)
    labels = mdace_label_space(processed)
    train_samples, val_samples = training_samples(args)

    # MDACE test admissions are only for evaluating explanations.
    test_ids = set(
        pl.read_parquet(processed / "mdace_icd9_inpatient" / "test.parquet")["_id"]
        .cast(pl.Utf8)
        .to_list()
    )
    train_ids = set(
        pl.read_parquet(processed / "mdace_icd9_inpatient_code" / "train.parquet")[
            "_id"
        ]
        .cast(pl.Utf8)
        .to_list()
    )
    leaked = test_ids & train_ids
    if leaked:
        raise ValueError(f"{len(leaked)} MDACE test admissions are in training.")

    print(
        f"labels {len(labels)}, train notes {len(train_samples)}, "
        f"val notes {len(val_samples)}"
    )

    extra = []
    if args.init_checkpoint:
        # One unused sample holding every code of the full train + val split
        # gives the starting model's label order, even with --max_train.
        vocab = set()
        for split in ("train", "val"):
            df = pl.read_parquet(
                processed / "mdace_icd9_inpatient_code" / f"{split}.parquet",
                columns=["diagnosis_codes", "procedure_codes"],
            )
            for row in df.iter_rows(named=True):
                vocab.update(note_codes(row))
        extra = [
            {
                "patient_id": "vocab",
                "note_id": "vocab",
                TEXT_KEY: "",
                LABEL_KEY: sorted(vocab & labels),
            }
        ]
    dataset = create_sample_dataset(
        samples=train_samples + val_samples + extra,
        input_schema={TEXT_KEY: "text"},
        output_schema={LABEL_KEY: "multilabel"},
        dataset_name="mdace_icd9_coding",
    )
    n_train = len(train_samples)
    train_ds = dataset.subset(range(n_train))
    val_ds = dataset.subset(range(n_train, n_train + len(val_samples)))

    torch.manual_seed(args.seed)
    model = build_model(args, dataset)
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    # Trainer is used for evaluation only. The training loop below follows
    # the authors' recipe, which Trainer lacks: gradient accumulation to an
    # effective batch of batch_size * accum, 10% linear warmup then linear
    # decay, and gradient clipping at 1. Without warmup the encoder collapsed
    # to predicting code frequencies.
    trainer = Trainer(
        model=model,
        metrics=["f1_micro", "f1_macro", "pr_auc_micro"],
        device=args.device,
        output_path=args.output_dir,
        exp_name=exp_name(args),
    )
    train_loader = get_dataloader(
        train_ds, args.batch_size, shuffle=True, seed=args.seed
    )
    val_loader = get_dataloader(val_ds, args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    updates = args.epochs * math.ceil(len(train_loader) / args.accum)
    warmup = max(1, int(args.warmup * updates))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 1.0  # TM: constant lr, as in the authors' TM config
        if args.tm
        else min((step + 1) / warmup, max(0.0, (updates - step) / (updates - warmup))),
    )
    if args.tm:
        # Token masking distils from the frozen starting model. Its logits are
        # computed on the fly (one no-grad forward per batch, small next to
        # the up to tm_iters mask updates) instead of the authors' pass over
        # the whole dataset before training.
        teacher = copy.deepcopy(model).eval().requires_grad_(False)
        mask_iters, kept = [], []
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}")):
            if args.tm:
                ids, mask = model.tokenize(batch[TEXT_KEY])
                with torch.no_grad():
                    teacher_logits = teacher.forward_from_embedding(
                        teacher.get_input_embeddings()(ids), mask
                    )["logit"]
                out = token_masking_loss(
                    model,
                    ids,
                    mask,
                    teacher_logits,
                    lambda_1=args.tm_lambda,
                    diet_constant=args.tm_diet,
                    lr=args.tm_mask_lr,
                    max_iter=args.tm_iters,
                    baseline_token_id=args.tm_baseline_id,
                    min_iter=args.tm_min_iters,
                    init=args.tm_mask_init,
                )
                mask_iters.append(out["iterations"])
                kept.append(out["mask"][mask.bool()].mean().item())
                loss = out["loss"] / args.accum
            else:
                loss = model(**batch)["loss"] / args.accum
            loss.backward()
            running += loss.item()
            last = step + 1 == len(train_loader)
            if (step + 1) % args.accum == 0 or last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if (step + 1) % (args.accum * 50) == 0:
                avg = running * args.accum / (args.accum * 50)
                print(f"step {step + 1}: train loss {avg:.4f}", flush=True)
                running = 0.0
        if args.tm:
            print(
                f"epoch {epoch} TM: mean mask iterations "
                f"{sum(mask_iters) / len(mask_iters):.2f}, tokens kept "
                f"{sum(kept) / len(kept):.1%}",
                flush=True,
            )
        scores = trainer.evaluate(val_loader)
        print(f"epoch {epoch} val: {scores}", flush=True)
        trainer.save_ckpt(os.path.join(trainer.exp_path, "last.ckpt"))
        if scores["pr_auc_micro"] > best:
            best = scores["pr_auc_micro"]
            trainer.save_ckpt(os.path.join(trainer.exp_path, "best.ckpt"))


def load_trained_model(args: argparse.Namespace) -> PLMCA | CAML:
    """Rebuilds the training label vocabulary and loads the checkpoint.

    PyHealth sorts the multilabel vocabulary, so one dummy sample holding
    every training code reproduces the training label order without
    re-processing the notes.
    """
    train_samples, val_samples = training_samples(args)
    vocab = sorted({c for s in train_samples + val_samples for c in s[LABEL_KEY]})
    dataset = create_sample_dataset(
        samples=[{"patient_id": "0", TEXT_KEY: "", LABEL_KEY: vocab}],
        input_schema={TEXT_KEY: "text"},
        output_schema={LABEL_KEY: "multilabel"},
        dataset_name="mdace_label_space",
    )
    model = build_model(args, dataset)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(args.device or "cpu").eval()


def explain_split(model: PLMCA | CAML, df: pl.DataFrame, steps: int) -> list[dict]:
    """Attention, InputXGrad and IG scores for every (note, gold code) pair.

    Special tokens are dropped and scores renormalized, as in the authors'
    evaluation. Pairs whose evidence was truncated away are skipped.
    """
    vocab = model.dataset.output_processors[LABEL_KEY].label_vocab
    explainers = {
        name: AttInGrad(model, method=name, steps=steps)
        for name in ("attention", "inputxgrad", "integrated_gradients")
    }
    pairs = []
    for row in tqdm(df.iter_rows(named=True), total=df.height):
        enc = model.tokenizer(
            row["text"],
            truncation=True,
            max_length=model.max_length,
            return_offsets_mapping=True,
        )
        offsets = enc["offset_mapping"]
        real = [i for i, (a, b) in enumerate(offsets) if b > a]
        position = {token: i for i, token in enumerate(real)}
        code_spans = list(
            zip(row["diagnosis_codes"] or [], row["diagnosis_code_spans"] or [])
        ) + list(zip(row["procedure_codes"] or [], row["procedure_code_spans"] or []))
        for code, spans in code_spans:
            if code not in vocab:
                continue
            gold = [position[t] for t in spans_to_token_ids(offsets, spans)]
            if not gold:
                continue
            pair = {"note_id": row["note_id"], "code": code, "evidence": gold}
            for name, explainer in explainers.items():
                scores = explainer.attribute(
                    **{TEXT_KEY: [row["text"]]}, target_class_idx=vocab[code]
                )[TEXT_KEY][0, real]
                pair[name] = (scores / scores.sum().clamp_min(1e-12)).cpu().numpy()
            pairs.append(pair)
    for pair in pairs:
        for product, part in (
            ("attingrad", "inputxgrad"),
            ("att_ig", "integrated_gradients"),
        ):
            joint = pair["attention"] * pair[part]
            pair[product] = joint / max(joint.sum(), 1e-12)
    return pairs


def evaluate(args: argparse.Namespace) -> None:
    model = load_trained_model(args)
    processed = Path(args.processed)
    pairs = {}
    for split in ("val", "test"):
        df = pl.read_parquet(processed / "mdace_icd9_inpatient" / f"{split}.parquet")
        # The paper evaluates discharge summaries only (authors' eval_explanations.py).
        df = df.filter(pl.col("note_type") == "Discharge summary")
        if args.max_eval_notes:
            df = df.head(args.max_eval_notes)
        pairs[split] = explain_split(model, df, args.ig_steps)
        print(f"{split}: {len(pairs[split])} (note, code) pairs")
    test_texts = dict(zip(df["note_id"], df["text"]))

    rows = []
    for method in EXPLANATION_METHODS:
        val_scores = [p[method] for p in pairs["val"]]
        threshold = best_plausibility_threshold(
            val_scores, [p["evidence"] for p in pairs["val"]]
        )
        test = plausibility_metrics(
            [p[method] for p in pairs["test"]],
            [p["evidence"] for p in pairs["test"]],
            threshold,
            ks=(5,),
        )
        rows.append({"method": method, "threshold": threshold, **test})

    if args.faithfulness:
        # Comprehensiveness / sufficiency as in the authors' evaluation code:
        # word-level (byte-level BPE tokenizers), mask-token baseline, K=100,
        # step 2, only pairs whose code is predicted (p > 0.5).
        from pyhealth.metrics.interpretability import faithfulness, word_feature_map

        vocab = model.dataset.output_processors[LABEL_KEY].label_vocab
        faith = {method: [] for method in EXPLANATION_METHODS}
        for pair in tqdm(pairs["test"], desc="faithfulness"):
            text = test_texts[pair["note_id"]]
            input_ids, attention_mask = model.tokenize(text)
            class_idx = vocab[pair["code"]]
            with torch.no_grad():
                embeds = model.get_input_embeddings()(input_ids)
                logit = model.forward_from_embedding(embeds, attention_mask)["logit"]
            if torch.sigmoid(logit[0, class_idx]) <= 0.5:
                continue
            offsets = model.tokenizer(
                text,
                truncation=True,
                max_length=model.max_length,
                return_offsets_mapping=True,
            )["offset_mapping"]
            positions = torch.tensor([i for i, (a, b) in enumerate(offsets) if b > a])
            tokens = model.tokenizer.convert_ids_to_tokens(
                input_ids[0, positions].tolist()
            )
            byte_level = any(t.startswith("Ġ") for t in tokens)
            words = word_feature_map(tokens) if byte_level else None
            for method in EXPLANATION_METHODS:
                faith[method].append(
                    faithfulness(
                        model,
                        input_ids[0],
                        attention_mask[0],
                        pair[method],
                        class_idx,
                        model.tokenizer.mask_token_id,
                        positions=positions,
                        feature_map=words,
                    )
                )
        n_pairs = len(faith[EXPLANATION_METHODS[0]])
        print(f"faithfulness: {n_pairs} test pairs with the code predicted")
        for row in rows:
            scores = faith[row["method"]]
            for key in ("comprehensiveness", "sufficiency"):
                values = [s[key] for s in scores]
                row[key] = sum(values) / len(values) if values else float("nan")
            row["faithfulness_pairs"] = n_pairs
    results = pl.DataFrame(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.write_csv(out_dir / "plausibility_test.csv")
    with pl.Config(tbl_rows=-1, float_precision=4):
        print(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=sorted(MODEL_DEFAULTS), default="plm_ca")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--ig_steps", type=int, default=20)
    parser.add_argument("--max_eval_notes", type=int, default=None)
    parser.add_argument(
        "--faithfulness",
        action="store_true",
        help="with --evaluate: add comprehensiveness and sufficiency (slow: "
        "about 102 forward passes per method and predicted test pair)",
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--processed", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output/mdace")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="2 (1 with --tm)")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--accum", type=int, default=None)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    # Token masking (TM) fine-tuning; defaults are the authors' TM config.
    parser.add_argument(
        "--tm",
        action="store_true",
        help="fine-tune --init_checkpoint with token masking; saves to "
        "{model}_tm_mdace (lr 1e-5 and 1 epoch unless set)",
    )
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--tm_lambda", type=float, default=1.0)
    parser.add_argument("--tm_diet", type=float, default=0.01)
    parser.add_argument("--tm_mask_lr", type=float, default=0.1)
    parser.add_argument("--tm_iters", type=int, default=20)
    parser.add_argument(
        "--tm_min_iters",
        type=int,
        default=0,
        help="mask updates before the convergence check may stop (authors: 0, "
        "which stops at update 3 with every token still kept)",
    )
    parser.add_argument(
        "--tm_mask_init", type=float, default=2.0, help="starting mask logit (authors: 2.0)"
    )
    parser.add_argument(
        "--tm_baseline_id",
        type=int,
        default=None,
        help="token id for masked tokens (default: the mask token; the "
        "authors' TM runs used 50001, an unused RoBERTa-PM embedding row)",
    )
    args = parser.parse_args()
    lr, batch_size, accum = MODEL_DEFAULTS[args.model]
    if args.tm:
        lr = 1e-5
        if not (args.synthetic or args.init_checkpoint):
            parser.error("--tm needs --init_checkpoint (the model to fine-tune)")
    if args.epochs is None:
        args.epochs = 1 if args.tm else 2
    args.lr = lr if args.lr is None else args.lr
    args.batch_size = batch_size if args.batch_size is None else args.batch_size
    args.accum = accum if args.accum is None else args.accum
    if not args.synthetic and (args.train or args.evaluate):
        if not (args.processed and args.model_name):
            parser.error("--processed and --model_name are required with real data")
        if args.evaluate and not args.checkpoint:
            parser.error("--evaluate needs --checkpoint")
    return args


def main() -> None:
    args = parse_args()
    if args.synthetic:
        run_synthetic(args)
        return
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)


if __name__ == "__main__":
    main()
