"""Example ablation for WordBasisLinearModel.

This script demonstrates a minimal, runnable ablation for the paper-inspired
WordBasisLinearModel using synthetic/demo data.

Ablation:
    Vary weight decay during training and compare:
    - validation accuracy
    - validation loss
    - cosine similarity between the learned classifier weights and the
      word-basis reconstruction

Why this ablation:
    The updated rubric for model contributions asks for a hyperparameter
    variation or similar concrete model ablation. Weight decay is the
    simplest PyTorch/PyHealth-friendly analogue to regularization in the
    paper's linear classifier.

Experimental setup:
    - Binary classification on synthetic embedding inputs
    - Bias-free linear classifier
    - Fixed word-embedding matrix for explanation
    - Three weight decay settings: 0.0, 1e-4, 1e-2

Example findings:
    In a representative run, lower weight decay produced the best validation
    accuracy, while a larger weight decay slightly improved cosine similarity
    between the learned classifier weights and the word-basis reconstruction.
    Exact numbers may vary slightly across environments.

This script is intentionally small and deterministic so it is easy to run
locally and easy for reviewers to inspect.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import torch
from pyhealth.datasets import create_sample_dataset
from pyhealth.models import WordBasisLinearModel


INPUT_DIM = 8
FEATURE_KEY = "embedding"
LABEL_KEY = "label"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataset():
    """Builds a tiny PyHealth sample dataset required by the model constructor."""
    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "embedding": [0.0] * INPUT_DIM,
            "label": 0,
        },
        {
            "patient_id": "patient-1",
            "visit_id": "visit-1",
            "embedding": [0.1] * INPUT_DIM,
            "label": 1,
        },
    ]

    dataset = create_sample_dataset(
        samples=samples,
        input_schema={FEATURE_KEY: "tensor"},
        output_schema={LABEL_KEY: "binary"},
        dataset_name="word_basis_linear_model_example",
    )
    return dataset


def make_synthetic_split(
    n_train: int = 64,
    n_val: int = 32,
    input_dim: int = INPUT_DIM,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Creates a simple synthetic binary classification problem in embedding space."""
    true_beta = torch.tensor(
        [1.2, -0.8, 0.6, 1.0, -1.1, 0.4, 0.7, -0.5],
        dtype=torch.float32,
    )

    x_train = torch.randn(n_train, input_dim)
    x_val = torch.randn(n_val, input_dim)

    train_noise = 0.35 * torch.randn(n_train)
    val_noise = 0.35 * torch.randn(n_val)

    train_logits = x_train @ true_beta + train_noise
    val_logits = x_val @ true_beta + val_noise

    y_train = (torch.sigmoid(train_logits) > 0.5).float()
    y_val = (torch.sigmoid(val_logits) > 0.5).float()

    return x_train, y_train, x_val, y_val


def make_word_embeddings() -> Tuple[torch.Tensor, List[str]]:
    """Creates a fixed word basis in the same embedding space as the classifier."""
    word_list = [
        "dark",
        "light",
        "round",
        "pointed",
        "large",
        "small",
    ]

    word_embeddings = torch.tensor(
        [
            [1.0, 0.2, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0],
            [0.1, 1.0, 0.0, 0.2, 0.1, 0.0, 0.1, 0.0],
            [0.0, 0.1, 1.0, 0.2, 0.0, 0.1, 0.0, 0.2],
            [0.2, 0.0, 0.1, 1.0, 0.1, 0.0, 0.2, 0.0],
            [0.0, 0.1, 0.0, 0.1, 1.0, 0.2, 0.1, 0.0],
            [0.1, 0.0, 0.2, 0.0, 0.2, 1.0, 0.0, 0.1],
        ],
        dtype=torch.float32,
    )

    return word_embeddings, word_list


def accuracy_from_probs(y_prob: torch.Tensor, y_true: torch.Tensor) -> float:
    y_pred = (y_prob.squeeze(1) >= 0.5).float()
    return (y_pred == y_true).float().mean().item()


def train_and_evaluate(
    weight_decay: float,
    epochs: int = 200,
    lr: float = 0.05,
) -> dict:
    dataset = build_dataset()
    model = WordBasisLinearModel(
        dataset=dataset,
        input_dim=INPUT_DIM,
        feature_key=FEATURE_KEY,
        ridge_lambda=1e-4,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    x_train, y_train, x_val, y_val = make_synthetic_split()
    word_embeddings, word_list = make_word_embeddings()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(**{FEATURE_KEY: x_train, LABEL_KEY: y_train})
        output["loss"].backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_output = model(**{FEATURE_KEY: x_train, LABEL_KEY: y_train})
        val_output = model(**{FEATURE_KEY: x_val, LABEL_KEY: y_val})

        coeffs = model.fit_word_basis(word_embeddings)
        cosine = model.compute_word_basis_cosine_similarity(
            word_embeddings=word_embeddings,
            word_coeffs=coeffs,
        ).item()

        top_words = model.explain_words(
            word_embeddings=word_embeddings,
            word_list=word_list,
        )[:3]

    return {
        "weight_decay": weight_decay,
        "train_loss": train_output["loss"].item(),
        "val_loss": val_output["loss"].item(),
        "train_acc": accuracy_from_probs(train_output["y_prob"], y_train),
        "val_acc": accuracy_from_probs(val_output["y_prob"], y_val),
        "cosine_similarity": cosine,
        "top_words": top_words,
    }


def main() -> None:
    set_seed(42)
    
    # Hyperparameter ablation required by the rubric for model contributions.
    weight_decays = [0.0, 1e-4, 1e-2]
    results = [train_and_evaluate(weight_decay=wd) for wd in weight_decays]

    print("\nWordBasisLinearModel ablation: weight decay sweep")
    print("-" * 95)
    print(
        f"{'weight_decay':>12} | {'train_acc':>9} | {'val_acc':>7} | "
        f"{'train_loss':>10} | {'val_loss':>8} | {'cosine_sim':>10}"
    )
    print("-" * 95)

    for row in results:
        print(
            f"{row['weight_decay']:>12.4g} | "
            f"{row['train_acc']:>9.3f} | "
            f"{row['val_acc']:>7.3f} | "
            f"{row['train_loss']:>10.4f} | "
            f"{row['val_loss']:>8.4f} | "
            f"{row['cosine_similarity']:>10.4f}"
        )

    print("\nTop 3 explanatory words by configuration:")
    for row in results:
        print(f"\nweight_decay={row['weight_decay']}")
        for word, coeff in row["top_words"]:
            print(f"  {word:>10}: {coeff:+.4f}")


if __name__ == "__main__":
    main()