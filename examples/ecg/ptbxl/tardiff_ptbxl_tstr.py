"""TarDiff on PTB-XL heartbeats: train on synthetic, test on real.

Reproduces the evaluation of TarDiff (Deng et al., "TarDiff: Target-Oriented
Diffusion Guidance for Synthetic Electronic Health Record Time Series
Generation", arXiv:2504.17613, 2025) on PTB-XL, with the paper's heartbeat
setup for PTB (MI vs healthy, single heartbeats; see
:class:`~pyhealth.tasks.PTBXLHeartbeatMI`).

Steps, for each seed:

1. Train a classifier on the real training folds (1-8) and test it on fold 10
   ("real": train real, test real).
2. Train TarDiff on the real training folds, with the noise schedule of
   TarDiff's official code (Latent Diffusion's scaled-linear betas from 5e-4
   to 0.1 over 200 steps). The diffusion model is trained once and saved to
   ``--diffusion-checkpoint``, by default a file named after the data and the
   training settings; later seeds and runs with the same settings reuse it.
3. Compute the guidance gradient ``G`` of the step-1 classifier over the
   validation fold (9), the paper's guidance set.
4. Generate as many synthetic heartbeats as there are real training beats,
   with the real class balance: unguided, and TarDiff-guided at each
   ``--guidance-scales``. ``--guidance official`` (the default) guides as
   TarDiff's official code does, scoring the clean-series estimate with a
   cosine; ``paper`` uses the paper's equation (see ``TarDiff.generate``).
5. TSTR: train a fresh classifier on each synthetic set alone. TSRTR: train
   on the real set plus synthetic beats at each ``--ratios`` of its size.
   Every classifier picks its best epoch on the real validation fold and is
   scored on the real test fold (AUROC, AUPRC).
6. With several guidance scales, ``tardiff_selected`` reports, for each
   setting and seed, the scale whose classifier did best on the validation
   fold (AUROC), so the scale isn't picked on the test fold.

The paper's downstream model is TimesNet, which PyHealth doesn't have; this
uses SparcNet for the classifier and the guide. Results go to
``<output>/results.json`` and a mean (std) table is printed.

Usage:
    # PTB-XL 1.0.3 from https://physionet.org/content/ptb-xl/1.0.3/
    python examples/ecg/ptbxl/tardiff_ptbxl_tstr.py --root /data/ptb-xl/1.0.3

    # About an hour on a laptop: a small U-Net, one beat per record.
    python examples/ecg/ptbxl/tardiff_ptbxl_tstr.py --root /data/ptb-xl/1.0.3 \\
        --unet small --max-beats-per-record 1 --iterations 5000 \\
        --diffusion-batch-size 128 --steps 25 --ratios 1.0 --epochs 5

    # A few-minute check on generated toy beats, no download:
    python examples/ecg/ptbxl/tardiff_ptbxl_tstr.py --synthetic --tiny
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pyhealth.datasets import (
    PTBXLDataset,
    create_sample_dataset,
    get_dataloader,
    split_by_strat_fold,
)
from pyhealth.models import SparcNet, TarDiff
from pyhealth.tasks import PTBXLHeartbeatMI
from pyhealth.trainer import Trainer

SCHEMA = {"input_schema": {"signal": "tensor"}, "output_schema": {"label": "binary"}}
# U-Nets: the paper's (its Appendix D; TarDiff's defaults, 16.3M parameters),
# a 1.2M-parameter one for laptop runs, and one for --tiny checks.
UNETS = {
    "paper": {},
    "small": {
        "base_channels": 32,
        "mults": (1, 2, 2, 2),
        "num_res_blocks": 2,
        "attention_levels": (2,),
        "heads": 4,
    },
    "tiny": {
        "base_channels": 16,
        "mults": (1, 2, 2, 2),
        "num_res_blocks": 1,
        "attention_levels": (2,),
        "heads": 2,
    },
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", help="PTB-XL 1.0.3 folder (with ptbxl_database.csv).")
    p.add_argument("--synthetic", action="store_true", help="Use toy beats instead.")
    p.add_argument("--unet", choices=sorted(UNETS), default="paper")
    p.add_argument("--tiny", action="store_true", help="Tiny U-Net and short runs.")
    p.add_argument("--output", default="output/tardiff_ptbxl")
    p.add_argument(
        "--cache-dir", default=str(Path.home() / ".cache/pyhealth/tardiff_ptbxl")
    )
    p.add_argument(
        "--diffusion-checkpoint",
        help="Default: a file in <output> named after the data and settings.",
    )
    p.add_argument(
        "--max-beats-per-record",
        type=int,
        default=4,
        help="4 gives ~58k beats, near the paper's 64k PTB beats.",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--iterations", type=int, default=20000, help="Paper: 20,000.")
    p.add_argument("--diffusion-batch-size", type=int, default=256, help="Paper: 256.")
    p.add_argument("--steps", type=int, default=50, help="DDIM sampling steps.")
    p.add_argument(
        "--beta-schedule",
        choices=["scaled_linear", "linear"],
        default="scaled_linear",
        help="Official code: scaled_linear (Latent Diffusion's 'linear').",
    )
    p.add_argument("--n-steps", type=int, default=200, help="Official code: 200.")
    p.add_argument("--beta-start", type=float, default=5e-4, help="Official: 5e-4.")
    p.add_argument("--beta-end", type=float, default=0.1, help="Official: 0.1.")
    p.add_argument(
        "--prediction",
        choices=["v", "eps"],
        default="v",
        help="Network target. The official code predicts eps; v is accurate "
        "at the noisiest steps (see TarDiff).",
    )
    p.add_argument(
        "--guidance-scales",
        type=float,
        nargs="+",
        default=None,
        help="Default 300 1000 3000 for --guidance official (samples drift "
        "off scale above ~1e4) and the paper's 100 (Appendix D) for "
        "--guidance paper.",
    )
    p.add_argument("--guidance", choices=["official", "paper"], default="official")
    p.add_argument("--ratios", type=float, nargs="+", default=[0.2, 0.5, 1.0])
    p.add_argument("--epochs", type=int, default=10, help="Classifier epochs.")
    p.add_argument("--batch-size", type=int, default=256, help="Classifier batch.")
    p.add_argument(
        "--device",
        default="cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )
    args = p.parse_args()
    if not args.synthetic and not args.root:
        p.error("pass --root or --synthetic")
    if args.guidance_scales is None:
        official = args.guidance == "official"
        args.guidance_scales = [300.0, 1000.0, 3000.0] if official else [100.0]
    if args.tiny:
        args.unet = "tiny"
        args.iterations = min(args.iterations, 300)
        args.steps = min(args.steps, 10)
        args.epochs = min(args.epochs, 3)
    return args


def toy_beats(n_records=120, beats=4, length=288):
    """Toy 12-lead beats: an R spike, with an inverted T wave for "MI"."""
    rng = np.random.default_rng(0)
    t = np.arange(length)
    samples = []
    for r in range(n_records):
        label = r % 2
        for b in range(beats):
            shift = rng.integers(-6, 7)
            r_wave = np.exp(-((t - 144 - shift) ** 2) / 18.0)
            t_wave = (
                (-1 if label else 1) * 0.4 * np.exp(-((t - 200 - shift) ** 2) / 200.0)
            )
            gains = rng.uniform(0.5, 1.5, (12, 1))
            x = gains * (r_wave + t_wave) + 0.1 * rng.standard_normal((12, length))
            samples.append(
                {
                    "patient_id": f"p{r}",
                    "record_id": str(r),
                    "beat": b,
                    "signal": x.astype(np.float32),
                    "label": label,
                    "strat_fold": r // 2 % 10 + 1,  # both classes in every fold
                }
            )
    return create_sample_dataset(samples=samples, **SCHEMA, dataset_name="toy_beats")


def load_samples(args):
    if args.synthetic:
        return toy_beats()
    dataset = PTBXLDataset(root=args.root, sampling_rate=500, cache_dir=args.cache_dir)
    task = PTBXLHeartbeatMI(
        dataset.scp_statements_path,
        max_beats_per_record=args.max_beats_per_record,
        skip_partial_beats=True,
    )
    return dataset.set_task(task, num_workers=os.cpu_count() or 1)


def raw_samples(dataset, prefix):
    """Processed samples back to raw dicts, to mix with synthetic ones."""
    return [
        {
            "patient_id": f"{prefix}-{s['patient_id']}",
            "signal": s["signal"].numpy(),
            "label": round(float(s["label"].reshape(-1)[0])),
        }
        for s in dataset
    ]


def as_dataset(samples, like):
    return create_sample_dataset(
        samples=samples,
        **SCHEMA,
        input_processors=like.input_processors,
        output_processors=like.output_processors,
    )


def fit_and_test(train, val_loader, test_loader, args, seed):
    """Trains SparcNet on ``train``; scores it on the real val and test folds."""
    torch.manual_seed(seed)
    model = SparcNet(train)
    trainer = Trainer(
        model, metrics=["roc_auc", "pr_auc"], device=args.device, enable_logging=False
    )
    trainer.train(
        get_dataloader(train, args.batch_size, shuffle=True, seed=seed),
        val_dataloader=val_loader,
        epochs=args.epochs,
        monitor="roc_auc",
    )
    scores = trainer.evaluate(test_loader)
    return model, {
        "roc_auc": scores["roc_auc"],
        "pr_auc": scores["pr_auc"],
        "val_roc_auc": trainer.evaluate(val_loader)["roc_auc"],
    }


def checkpoint_name(args):
    """Names the diffusion checkpoint after everything that changes it."""
    data = "toy" if args.synthetic else f"ptbxl-b{args.max_beats_per_record}"
    return (
        f"tardiff-{data}-{args.unet}-{args.prediction}-{args.beta_schedule}"
        f"-T{args.n_steps}"
        f"-{args.beta_start:g}-{args.beta_end:g}"
        f"-it{args.iterations}-bs{args.diffusion_batch_size}.pt"
    )


def diffusion_model(train, args):
    model = TarDiff(
        train,
        **UNETS[args.unet],
        n_steps=args.n_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        prediction=args.prediction,
    ).to(args.device)
    path = Path(args.diffusion_checkpoint or Path(args.output) / checkpoint_name(args))
    if path.exists():
        model.load_model(str(path))
        print(f"Loaded the diffusion model from {path}")
        if args.diffusion_checkpoint:
            print(
                "Warning: --diffusion-checkpoint was given, so only the model "
                "settings were checked, not the data it was trained on."
            )
        return model
    torch.manual_seed(0)
    loader = get_dataloader(train, args.diffusion_batch_size, shuffle=True, seed=0)
    losses = model.train_model(loader, iterations=args.iterations, progress=True)
    model.save_model(str(path))
    print(
        f"Trained the diffusion model (final loss {np.mean(losses[-100:]):.4f}); "
        f"saved to {path}"
    )
    return model


def generate(diffusion, labels, args, seed, name, kwargs, batch_size=256):
    """``diffusion.generate`` one batch at a time, with a progress bar.

    ``generate`` draws its batches in order from one random generator, so
    this gives the same samples as a single call.
    """
    gen = torch.Generator(device=args.device).manual_seed(seed)
    chunks = [
        diffusion.generate(
            labels[i : i + batch_size],
            steps=args.steps,
            batch_size=batch_size,
            generator=gen,
            **kwargs,
        )
        for i in tqdm(range(0, len(labels), batch_size), desc=f"Generating {name}")
    ]
    return {key: torch.cat([c[key] for c in chunks]) for key in ("signal", "label")}


def add_selected(scores, scales):
    """Adds ``tardiff_selected/*``: each setting at its best-validation scale."""
    settings = {k.split("/", 1)[1] for k in scores if k.startswith("tardiff_w")}
    for setting in settings:
        best = max(
            scales, key=lambda w: scores[f"tardiff_w{w:g}/{setting}"]["val_roc_auc"]
        )
        scores[f"tardiff_selected/{setting}"] = {
            **scores[f"tardiff_w{best:g}/{setting}"],
            "guidance_scale": best,
        }


def summarize(results):
    keys = sorted({k for run in results for k in run["scores"]})
    print(f"\n{'setting':28s} {'AUROC':>16s} {'AUPRC':>16s}")
    for key in keys:
        cells = []
        for metric in ("roc_auc", "pr_auc"):
            v = np.array([run["scores"][key][metric] for run in results])
            cells.append(f"{v.mean():.4f} ({v.std():.4f})")
        print(f"{key:28s} {cells[0]:>16s} {cells[1]:>16s}")


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args)
    train, val, test = split_by_strat_fold(samples)
    print(f"Heartbeats: {len(train)} train, {len(val)} val, {len(test)} test")
    val_loader = get_dataloader(val, args.batch_size)
    test_loader = get_dataloader(test, args.batch_size)
    real = raw_samples(train, "real")
    real_labels = [s["label"] for s in real]
    diffusion = diffusion_model(train, args)

    results = []
    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        guide, real_scores = fit_and_test(train, val_loader, test_loader, args, seed)
        scores = {"real": real_scores}
        G = TarDiff.guidance_gradient(guide, val_loader)
        labels = diffusion.class_indices(random.sample(real_labels, len(real_labels)))
        methods = {"unguided": {}}
        for w in args.guidance_scales:
            methods[f"tardiff_w{w:g}"] = {
                "guide": guide,
                "guidance": G,
                "guidance_scale": w,
                "guide_on_x0": args.guidance == "official",
                "cosine": args.guidance == "official",
            }
        for name, kwargs in methods.items():
            generated = generate(diffusion, labels, args, seed, name, kwargs)
            synthetic = diffusion.to_samples(generated, prefix=f"{name}-{seed}")
            _, scores[f"{name}/tstr"] = fit_and_test(
                as_dataset(synthetic, train), val_loader, test_loader, args, seed
            )
            for r in args.ratios:
                mixed = real + synthetic[: int(r * len(real))]
                _, scores[f"{name}/tsrtr_{r:g}"] = fit_and_test(
                    as_dataset(mixed, train), val_loader, test_loader, args, seed
                )
        add_selected(scores, args.guidance_scales)
        results.append({"seed": seed, "scores": scores})
        (out / "results.json").write_text(
            json.dumps({"args": vars(args), "results": results}, indent=2)
        )
        print(f"seed {seed}: {json.dumps(scores)}")
    summarize(results)


if __name__ == "__main__":
    main()
