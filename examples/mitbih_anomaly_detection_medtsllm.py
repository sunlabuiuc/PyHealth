"""MIT-BIH anomaly detection with MedTsLLM: full pipeline and ablations.

Contributors: Thi Van Trinh Tran and Zexi Li.
Paper: MedTsLLM, https://arxiv.org/abs/2408.07773.
Data: https://physionet.org/content/mitdb/1.0.0/ (open access, no credentials).

What it shows:
    MITBIHDataset -> MITBIHAnomalyDetection -> MedTsLLMAnomaly trained with
    the PyHealth Trainer -> point-wise anomaly scores and metrics, on the
    paper's record split (6 least / 4 most abnormal MLII+V1 records). The
    model only learns to reconstruct windows; labels are used for evaluation.

How to run:
    Synthetic (default, CPU, about a minute; tiny random GPT-2, no download)::

        python examples/mitbih_anomaly_detection_medtsllm.py

    Real data (download mitdb 1.0.0 from PhysioNet; a GPU is needed for
    Llama-2-7B, which requires accepting Meta's license on Hugging Face)::

        python examples/mitbih_anomaly_detection_medtsllm.py \\
            --root /data/mitdb/1.0.0 --llm meta-llama/Llama-2-7b-hf \\
            --epochs 10 --device cuda

    ``--llm gpt2`` runs a smaller real backbone, ``--random-backbone`` keeps
    the LLM architecture with random (frozen) weights.

Setup (paper configuration, ``configs/datasets/ecgmit-anom.toml`` of the
authors' code): 125 Hz, 128-step non-overlapping windows, both leads
concatenated, patches of 16 with stride 8, d_model 32, d_ff 64, 8 heads,
1024 text prototypes, prompt with dataset and task description, Adam with
learning rate 1e-4, batch size 16, 10 epochs, MSE loss, channels
standardized with training-set statistics.

Ablations (evaluation-time, one trained model):
    1. Evaluation protocol. The paper's threshold flags the top
       ``contamination`` fraction of scores, with the contamination computed
       from the *test* labels ("auto"), and then applies point adjustment
       (one hit marks a whole true anomalous segment as detected). We compare
       it with a threshold from the *training* labels and without point
       adjustment.
    2. Label definition (task configuration). Abnormal beats are expanded by
       ``half_window_ms`` on each side (paper: 150 ms). Training never sees
       labels, so the same model is scored against labels built with 50, 150
       and 300 ms.

Results:
    Synthetic data (default command: tiny random GPT-2, 3 epochs, seed 0, CPU,
    about a minute; it checks that the pipeline runs end to end, and the
    numbers say nothing about real ECGs). AUROC is computed on the binary
    predictions as in the paper; "score AUROC" is the threshold-free AUROC of
    the raw scores::

        threshold  point adjust   F1     AUROC   score AUROC
        auto       yes            0.957  0.994   0.754   <- paper protocol
        auto       no             0.443  0.649   0.754
        train      yes            0.959  0.995   0.754
        train      no             0.372  0.619   0.754

        half_window_ms   F1     AUROC   score AUROC   (auto, point adjust)
        50               1.000  1.000   0.996
        150              0.957  0.994   0.754         <- paper labels
        300              0.955  0.987   0.802

    Findings: point adjustment, not the threshold choice, makes up most of
    the F1 under the paper's protocol (0.44 -> 0.96 here), while the
    threshold-free score AUROC is unaffected.

    Real data, this script with ``--root <mitdb> --llm gpt2 --epochs 10
    --device cuda`` (Colab GPU, seed 0)::

        threshold  point adjust   F1     AUROC   score AUROC
        auto       yes            0.959  0.984   0.801   <- paper protocol
        auto       no             0.452  0.643   0.801
        train      yes            0.062  0.516   0.801
        train      no             0.002  0.500   0.801

        half_window_ms   F1     AUROC   score AUROC   (auto, point adjust)
        50               0.895  0.953   0.877
        150              0.959  0.984   0.801         <- paper labels
        300              0.967  0.962   0.739

    Findings on real data: under the paper's protocol GPT-2 matches the
    paper's Llama-2-7B result (F1 94.70, AUROC 98.52). Point adjustment again
    makes up most of the F1. A threshold from the training labels flags
    almost nothing, because the paper's split trains on the least abnormal
    records, so the "auto" threshold depends on the test labels' anomaly
    rate. The threshold-free score AUROC (0.80) is the fairer summary.

    Llama-2-7B. With the authors' code on our preprocessing of the same split
    (paper configuration, Colab A100, run 2026-11-03-03-12-40): F1 95.81,
    AUROC 98.36. Loading those trained weights into MedTsLLMAnomaly and
    evaluating with this task and anomaly_detection_metrics gives F1 95.83,
    AUROC 98.38 (0.448 without point adjustment, score AUROC 0.782). Given
    the same weights and inputs, outputs, gradients and metrics are identical
    to the authors' code (checked with tiny Llama and GPT-2 backbones, with
    and without patient prompts). The task's signals and labels match the
    files of that run (signals within 3e-6, no label mismatches); the only
    difference is that the authors' loader cuts windows from the
    concatenated records, while this task windows each record.
"""

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from pyhealth.datasets import MITBIHDataset, get_dataloader
from pyhealth.metrics import anomaly_detection_metrics, reconstruction_anomaly_scores
from pyhealth.models import MedTsLLMAnomaly
from pyhealth.tasks import MITBIHAnomalyDetection
from pyhealth.trainer import Trainer

# The authors' MIT-BIH description, used as the "Dataset:" prompt.
MITBIH_DESCRIPTION = (
    "The MIT-BIH Arrhythmia Database contains excerpts of two-channel ambulatory "
    "ECG from a mixed population of inpatients and outpatients, digitized at 360 "
    "samples per second per channel with 11-bit resolution over a 10 mV range."
)
FS = 360


class _CharTokenizer:
    """Offline character tokenizer for the tiny synthetic backbone only."""

    bos_token, eos_token, pad_token, pad_token_id = "<s>", "</s>", None, 2

    def __call__(self, text, **kwargs):
        ids = [1] + ([1] if text == "<s>" else [3 + ord(c) % 60 for c in text])
        return SimpleNamespace(input_ids=torch.tensor([ids]))


def write_synthetic_records(root: Path, seconds: int = 20, seed: int = 0) -> None:
    """Writes 10 two-lead MLII+V1 WFDB records with normal and abnormal beats.

    Normal beats have a P wave, a narrow QRS and a T wave; abnormal beats
    ("V") are wide, inverted and have no P wave. Record ``i`` has ``i``
    abnormal beats in 20, so the paper split trains on the least abnormal.
    """
    import wfdb

    rng = np.random.default_rng(seed)
    t = np.arange(seconds * FS) / FS

    def bump(center, width, amp):
        return amp * np.exp(-0.5 * ((t - center) / width) ** 2)

    for i in range(10):
        beats = np.arange(0.5, seconds - 0.5, seconds / 20)[:20]
        abnormal = set(rng.choice(20, size=i, replace=False).tolist())
        signal = np.zeros_like(t)
        symbols = []
        for k, b in enumerate(beats):
            if k in abnormal:
                signal += bump(b, 0.04, -1.5) + bump(b + 0.25, 0.08, 0.4)
                symbols.append("V")
            else:
                signal += bump(b - 0.16, 0.025, 0.15) + bump(b, 0.012, 1.0)
                signal += bump(b + 0.3, 0.05, 0.3)
                symbols.append("N")
        leads = np.stack([signal, -0.6 * signal + 0.1], axis=1)
        leads += 0.02 * rng.standard_normal(leads.shape)
        name = str(100 + i)
        wfdb.wrsamp(
            name,
            fs=FS,
            units=["mV", "mV"],
            sig_name=["MLII", "V1"],
            p_signal=leads,
            fmt=["212", "212"],
            comments=[f"{50 + i} {'MF'[i % 2]}"],
            write_dir=str(root),
        )
        wfdb.wrann(
            name, "atr", np.round(beats * FS).astype(int), symbols, write_dir=str(root)
        )


def label_ratio(dataset) -> tuple[int, int]:
    """Number of anomalous and of all time steps in a step-labeled set."""
    labels = np.concatenate([np.asarray(s["label"]) for s in dataset])
    return int(labels.sum()), int(labels.size)


def window_keys(dataset) -> list[tuple[str, int]]:
    """(record, start) of each window, in dataset order."""
    return [(s["record_id"], int(s["window_start"])) for s in dataset]


def test_scores(trainer: Trainer, test, batch_size: int) -> dict:
    """Per-step anomaly scores of each test window, keyed by (record, start)."""
    loader = get_dataloader(test, batch_size=batch_size, shuffle=False)
    y_true, y_prob, _ = trainer.inference(loader)
    keys = window_keys(test)
    y_true = np.asarray(y_true).reshape(len(keys), -1, test[0]["signal"].shape[-1])
    y_prob = np.asarray(y_prob).reshape(y_true.shape)
    # normalize_by_feature divides by the mean error over all test steps.
    scores = reconstruction_anomaly_scores(y_prob, y_true, normalize_by_feature=True)
    return dict(zip(keys, scores.reshape(len(keys), -1)))


def in_time_order(per_window: dict) -> np.ndarray:
    """Concatenates per-window arrays by record, then time (for point adjust)."""
    return np.concatenate([per_window[k] for k in sorted(per_window)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, help="mitdb 1.0.0 (default: synthetic)")
    parser.add_argument("--llm", help="Hugging Face LLM (default: tiny random GPT-2)")
    parser.add_argument("--random-backbone", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="write the results as JSON")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        root = args.root
        if root is None:
            root = tmp / "mitdb"
            root.mkdir()
            write_synthetic_records(root, seed=args.seed)

        # 1. Dataset and the paper's record split.
        dataset = MITBIHDataset(
            root=root, cache_dir=tmp / "cache", metadata_cache_dir=tmp / "meta"
        )
        split = dataset.anomaly_ratio_split()
        print("train/test records:", split.select_records())

        # 2. Task: 125 Hz, 128-step windows, abnormal beats +/-150 ms.
        task = MITBIHAnomalyDetection()
        train, test = dataset.set_task(task, split=split)

        # 3. Model: frozen LLM with trainable MedTsLLM adapters.
        if args.llm:
            model = MedTsLLMAnomaly(
                train,
                llm_name=args.llm,
                random_backbone=args.random_backbone,
                dataset_description=MITBIH_DESCRIPTION,
            )
        else:
            from transformers import GPT2Config, GPT2Model

            tiny = GPT2Model(
                GPT2Config(
                    vocab_size=64, n_embd=32, n_layer=2, n_head=4, n_positions=512
                )
            )
            model = MedTsLLMAnomaly(
                train,
                backbone=tiny,
                tokenizer=_CharTokenizer(),
                num_tokens=64,
                dataset_description=MITBIH_DESCRIPTION,
            )

        # 4. Training: Adam, lr 1e-4, MSE reconstruction (the model's loss).
        trainer = Trainer(
            model, device=args.device, metrics=["mse"], enable_logging=False
        )
        trainer.train(
            get_dataloader(train, args.batch_size, shuffle=True),
            epochs=args.epochs,
            optimizer_params={"lr": 1e-4},
            load_best_model_at_last=False,
        )

        # 5. Evaluation and ablations.
        scores = in_time_order(test_scores(trainer, test, args.batch_size))
        n_train_anom, n_train = label_ratio(train)
        results = []

        def evaluate(test_set, name, threshold, adjust):
            labels = in_time_order(
                dict(
                    zip(
                        window_keys(test_set),
                        (np.asarray(s["label"]) for s in test_set),
                    )
                )
            )
            if threshold == "auto":  # the authors' choice: uses test labels
                contamination = labels.sum() / (labels.size + n_train)
            else:  # from the training labels only
                contamination = n_train_anom / n_train
            m = anomaly_detection_metrics(scores, labels, contamination, adjust)
            results.append(
                {
                    "ablation": name,
                    "threshold_from": threshold,
                    "point_adjust": adjust,
                    **m,
                }
            )
            print(
                f"{name:28s} threshold={threshold:5s} point_adjust={adjust!s:5s} "
                f"F1 {m['f1']:.3f}  AUROC {m['roc_auc']:.3f}  "
                f"(score AUROC {m['score_roc_auc']:.3f})"
            )

        print("\nAblation 1: evaluation protocol (paper: auto + point adjust)")
        for threshold in ("auto", "train"):
            for adjust in (True, False):
                evaluate(test, "protocol", threshold, adjust)

        print("\nAblation 2: anomaly label width (paper: 150 ms)")
        for half_window_ms in (50.0, 150.0, 300.0):
            _, test_h = dataset.set_task(
                MITBIHAnomalyDetection(half_window_ms=half_window_ms), split=split
            )
            evaluate(test_h, f"labels half_window_ms={half_window_ms:g}", "auto", True)

        if args.output:
            args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
