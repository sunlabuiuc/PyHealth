import argparse
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from pyhealth.datasets import PTBXLDataset, get_dataloader, split_by_strat_fold
from pyhealth.models import PTBXLResNet18
from pyhealth.tasks import PTBXLSuperclassClassification


def normalize_batch(batch, mean, std):
    batch = dict(batch)
    batch["signal"] = (
        batch["signal"] - mean[None, :, None]
    ) / std[None, :, None]
    return batch


def lead_statistics(dataset, batch_size):
    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)
    total = torch.zeros(12, dtype=torch.float64)
    total_sq = torch.zeros(12, dtype=torch.float64)
    count = 0

    for batch in tqdm(loader, desc="Computing training-lead statistics"):
        x = batch["signal"].to(torch.float64)
        total += x.sum(dim=(0, 2))
        total_sq += (x * x).sum(dim=(0, 2))
        count += x.shape[0] * x.shape[2]

    mean = total / count
    variance = (total_sq / count - mean * mean).clamp_min(1e-12)
    return mean.float(), variance.sqrt().float()


def evaluate(model, loader, mean, std):
    model.eval()
    losses, targets, probabilities = [], [], []

    with torch.no_grad():
        for batch in loader:
            result = model(**normalize_batch(batch, mean, std))
            n = batch["signal"].shape[0]
            losses.append(float(result["loss"]) * n)
            targets.append(result["y_true"].cpu().numpy())
            probabilities.append(result["y_prob"].cpu().numpy())

    y_true = np.concatenate(targets)
    y_prob = np.concatenate(probabilities)
    valid = [
        i for i in range(y_true.shape[1])
        if np.unique(y_true[:, i]).size == 2
    ]
    auc = (
        roc_auc_score(y_true[:, valid], y_prob[:, valid], average="macro")
        if valid else float("nan")
    )
    return sum(losses) / len(y_true), auc, len(valid)


def run_schedule(name, args, samples, train, val, mean, std):
    torch.manual_seed(args.seed)
    train_loader = get_dataloader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val, batch_size=args.batch_size, shuffle=False)

    model = PTBXLResNet18(samples)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        if name == "cosine" else None
    )

    best_auc = -float("inf")
    best_epoch = 0
    best_state = None
    start = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        total_loss, count = 0.0, 0

        for batch in tqdm(
            train_loader, desc=f"{name}: epoch {epoch + 1}/{args.epochs}"
        ):
            batch = normalize_batch(batch, mean, std)
            optimizer.zero_grad(set_to_none=True)
            result = model(**batch)
            result["loss"].backward()
            optimizer.step()
            n = batch["signal"].shape[0]
            total_loss += result["loss"].detach().item() * n
            count += n

        if scheduler is not None:
            scheduler.step()

        val_loss, val_auc, valid_labels = evaluate(
            model, val_loader, mean, std
        )
        print(
            f"{name} epoch {epoch + 1}: "
            f"train_loss={total_loss / count:.4f}, "
            f"val_loss={val_loss:.4f}, val_macro_auc={val_auc:.4f} "
            f"({valid_labels} labels)"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    print(f"{name} training time: {(time.perf_counter() - start) / 60:.1f} min")
    return model, best_epoch, best_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path.home() / "ptbxl-data" / "ptbxl"
    )
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(4)
    dataset = PTBXLDataset(
        root=str(args.root), sampling_rate=500, dev=args.dev
    )
    task = PTBXLSuperclassClassification(
        scp_statements_path=str(args.root / "scp_statements.csv")
    )
    samples = dataset.set_task(task)
    train, val, test = split_by_strat_fold(samples)
    print(
        f"Samples: {len(samples)} | Train: {len(train)} | "
        f"Val: {len(val)} | Test: {len(test)}"
    )
    if not len(train) or not len(val) or not len(test):
        raise RuntimeError("A split is empty; stop before training.")

    mean, std = lead_statistics(train, args.batch_size)
    test_loader = get_dataloader(test, batch_size=args.batch_size, shuffle=False)

    results = {}
    for schedule in ("constant", "cosine"):
        model, best_epoch, best_val_auc = run_schedule(
            schedule, args, samples, train, val, mean, std
        )
        results[schedule] = (model, best_epoch, best_val_auc)

    print("\nTest results from each run's best validation-AUROC checkpoint:")
    for schedule, (model, epoch, val_auc) in results.items():
        test_loss, test_auc, valid_labels = evaluate(
            model, test_loader, mean, std
        )
        print(
            f"{schedule}: best_epoch={epoch}, val_macro_auc={val_auc:.4f}, "
            f"test_loss={test_loss:.4f}, test_macro_auc={test_auc:.4f} "
            f"({valid_labels} labels)"
        )

    if args.dev:
        print(
            "Limited-data run: dev mode used; report this as a partial "
            "reproduction, not a full PTB-XL reproduction."
        )


if __name__ == "__main__":
    main()
