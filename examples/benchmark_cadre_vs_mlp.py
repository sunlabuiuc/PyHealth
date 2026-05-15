import random
import time
from typing import Dict, List

import numpy as np
import torch
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import CADRE, MLP
from sklearn.metrics import f1_score


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def build_dataset(
    n_samples: int = 60,
    seq_len: int = 8,
    num_genes: int = 50,
    num_labels: int = 2,
):
    """Create a small synthetic multilabel dataset.

    The labels are intentionally correlated with simple gene-pattern rules so
    models have something learnable.
    """
    samples: List[Dict] = []

    for i in range(n_samples):
        genes = np.random.randint(1, num_genes + 1, size=seq_len).tolist()

        # Simple synthetic label rules:
        # label 0 active if more even genes than odd
        even_count = sum(g % 2 == 0 for g in genes)
        label_tokens = []
        if even_count >= seq_len // 2:
            label_tokens.append(0)

        # label 1 active if mean gene id is relatively high
        if np.mean(genes) > (num_genes / 2):
            label_tokens.append(1)

        # Guarantee at least one active label
        if not label_tokens:
            label_tokens = [0]

        samples.append(
            {
                "patient_id": f"patient-{i}",
                "visit_id": f"visit-{i}",
                "gene_idx": genes,
                "label": label_tokens,
            }
        )

    input_schema = {
        "gene_idx": "sequence",
    }
    output_schema = {
        "label": "multilabel",
    }

    dataset = create_sample_dataset(
        samples=samples,
        input_schema=input_schema,
        output_schema=output_schema,
        dataset_name="cadre_benchmark",
    )
    return dataset


def split_dataset(dataset, train_ratio: float = 0.8):
    """Deterministic split for reproducibility."""
    n = len(dataset)
    n_train = int(n * train_ratio)
    indices = list(range(n))
    train_ds = torch.utils.data.Subset(dataset, indices[:n_train])
    test_ds = torch.utils.data.Subset(dataset, indices[n_train:])
    return train_ds, test_ds


def evaluate_metrics(model, dataloader, device):
    """Compute average loss and micro F1."""
    model.eval()

    losses = []
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

            ret = model(**batch)

            losses.append(ret["loss"].item())

            y_true = ret["y_true"].cpu().numpy()
            y_prob = ret["y_prob"].cpu().numpy()

            y_pred = (y_prob >= 0.5).astype(int)

            y_true_all.append(y_true)
            y_pred_all.append(y_pred)

    y_true_all = np.vstack(y_true_all)
    y_pred_all = np.vstack(y_pred_all)

    f1 = f1_score(
        y_true_all,
        y_pred_all,
        average="micro",
        zero_division=0,
    )

    return float(np.mean(losses)), float(f1)


def train_model(model, train_loader, test_loader, device, epochs: int = 5, lr: float = 1e-3):
    """Simple training loop for quick benchmarking."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []

    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []

        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

            optimizer.zero_grad()
            ret = model(**batch)
            loss = ret["loss"]
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))
        test_loss, test_f1 = evaluate_metrics(model, test_loader, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(
            f"  epoch={epoch} train_loss={train_loss:.4f} "
            f"test_loss={test_loss:.4f} "
            f"test_f1={test_f1:.4f}"
        )

    elapsed = time.time() - start

    return {
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1],
        "best_test_loss": min(test_losses),
        "final_test_f1": test_f1,
        "runtime_sec": elapsed,
    }


def benchmark_cadre(use_attention: bool, train_loader, test_loader, dataset, device):
    """Benchmark one CADRE configuration."""
    model = CADRE(
        dataset=dataset,
        feature_key="gene_idx",
        label_key="label",
        num_genes=dataset.input_processors["gene_idx"].size(),
        num_drugs=2,
        embedding_dim=16,
        hidden_dim=16,
        attention_size=8,
        attention_head=2,
        dropout=0.1,
        use_attention=use_attention,
        use_cntx_attn=use_attention,
    )

    label = "CADRE(attention=True)" if use_attention else "CADRE(attention=False)"
    print(f"\n=== {label} ===")
    result = train_model(model, train_loader, test_loader, device, epochs=5, lr=1e-3)
    return label, result


def benchmark_mlp(train_loader, test_loader, dataset, device):
    """Benchmark MLP baseline."""
    model = MLP(
        dataset=dataset,
        embedding_dim=16,
        hidden_dim=16,
        n_layers=2,
    )

    print("\n=== MLP ===")
    result = train_model(model, train_loader, test_loader, device, epochs=5, lr=1e-3)
    return "MLP", result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    dataset = build_dataset()

    train_loader = get_dataloader(dataset, batch_size=8, shuffle=True)
    test_loader = get_dataloader(dataset, batch_size=8, shuffle=False)

    results = {}

    name, result = benchmark_cadre(
        use_attention=True,
        train_loader=train_loader,
        test_loader=test_loader,
        dataset=dataset,
        device=device,
    )
    results[name] = result

    name, result = benchmark_cadre(
        use_attention=False,
        train_loader=train_loader,
        test_loader=test_loader,
        dataset=dataset,
        device=device,
    )
    results[name] = result

    name, result = benchmark_mlp(
        train_loader=train_loader,
        test_loader=test_loader,
        dataset=dataset,
        device=device,
    )
    results[name] = result

    print("\n=== Benchmark Summary ===")
    for model_name, metrics in results.items():
        print(
            f"{model_name:24s} "
            f"final_test_loss={metrics['final_test_loss']:.4f} "
            f"final_test_f1={metrics['final_test_f1']:.4f} "
            f"runtime_sec={metrics['runtime_sec']:.2f}"
        )

    print("\nInterpretation:")
    print("- Lower test loss is better in this quick benchmark.")
    print("- This is a synthetic-data sanity benchmark, not a real-world claim.")
    print("- It shows that CADRE can be benchmarked against an existing PyHealth baseline.")


if __name__ == "__main__":
    main()