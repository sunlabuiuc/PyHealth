import random
import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import CADRE


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def build_dataset():
    """Create a tiny synthetic multilabel dataset for CADRE."""
    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "gene_idx": [1, 2, 3, 4, 0, 0],
            "label": [0],
        },
        {
            "patient_id": "patient-1",
            "visit_id": "visit-1",
            "gene_idx": [5, 6, 7, 8, 9, 0],
            "label": [1],
        },
        {
            "patient_id": "patient-2",
            "visit_id": "visit-2",
            "gene_idx": [2, 4, 6, 8, 0, 0],
            "label": [0],
        },
        {
            "patient_id": "patient-3",
            "visit_id": "visit-3",
            "gene_idx": [1, 3, 5, 7, 9, 0],
            "label": [1],
        },
    ]

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
        dataset_name="cadre_demo",
    )
    return dataset


def run_config(name: str, use_attention: bool):
    """Run one CADRE configuration and print summary stats."""
    dataset = build_dataset()
    loader = get_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    model = CADRE(
        dataset=dataset,
        feature_key="gene_idx",
        label_key="label",
        num_genes=20,
        num_drugs=2,
        embedding_dim=16,
        hidden_dim=16,
        attention_size=8,
        attention_head=2,
        dropout=0.1,
        use_attention=use_attention,
        use_cntx_attn=use_attention,
    )

    model.eval()
    with torch.no_grad():
        ret = model(**batch)

    print(f"\n=== {name} ===")
    print(f"use_attention={use_attention}")
    print(f"loss={ret['loss'].item():.6f}")
    print(f"logit shape={tuple(ret['logit'].shape)}")
    print(f"y_prob shape={tuple(ret['y_prob'].shape)}")
    print(f"mean y_prob={ret['y_prob'].mean().item():.6f}")

    return {
        "name": name,
        "use_attention": use_attention,
        "loss": ret["loss"].item(),
        "mean_y_prob": ret["y_prob"].mean().item(),
    }


def main():
    """Simple ablation: attention on vs off."""
    print("Running CADRE ablation on synthetic data...")
    print("Ablation: use_attention=True vs use_attention=False")

    result_attn = run_config("CADRE with attention", use_attention=True)
    result_no_attn = run_config("CADRE without attention", use_attention=False)

    print("\n=== Summary ===")
    print(
        f"with attention:    loss={result_attn['loss']:.6f}, "
        f"mean_y_prob={result_attn['mean_y_prob']:.6f}"
    )
    print(
        f"without attention: loss={result_no_attn['loss']:.6f}, "
        f"mean_y_prob={result_no_attn['mean_y_prob']:.6f}"
    )
    print(
        "\nThis example demonstrates a runnable CADRE ablation using "
        "synthetic multilabel data in PyHealth."
    )


if __name__ == "__main__":
    main()