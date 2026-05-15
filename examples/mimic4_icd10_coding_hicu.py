"""HiCu ICD-10 coding example with training experiments.

Runs in synthetic mode by default, or on real MIMIC-IV data with --data-dir.

Usage:
    python examples/mimic4_icd10_coding_hicu.py
    python examples/mimic4_icd10_coding_hicu.py --data-dir /path/to/mimic-iv
"""

import argparse
import torch
from pyhealth.datasets import MIMIC4Dataset, create_sample_dataset, get_dataloader
from pyhealth.models.hicu import HiCu
from pyhealth.tasks import MIMIC4ICD10Coding
from pyhealth.trainer import Trainer


def create_synthetic_dataset():
    """Create a small synthetic ICD-10 multilabel dataset."""
    samples = [
        {
            "patient_id": "p0", "visit_id": "v0",
            "text": ["patient", "admitted", "with", "type", "two", "diabetes", "and", "hypertension"],
            "icd_codes": ["E11.321", "I10", "J44.1"],
        },
        {
            "patient_id": "p1", "visit_id": "v1",
            "text": ["chest", "pain", "shortness", "of", "breath", "elevated", "troponin"],
            "icd_codes": ["I21.09", "I11.0", "I10"],
        },
        {
            "patient_id": "p2", "visit_id": "v2",
            "text": ["abdominal", "pain", "nausea", "vomiting", "gastroesophageal", "reflux"],
            "icd_codes": ["K21.0", "E11.65"],
        },
        {
            "patient_id": "p3", "visit_id": "v3",
            "text": ["chronic", "obstructive", "pulmonary", "disease", "exacerbation", "with", "respiratory", "failure"],
            "icd_codes": ["J44.1", "E11.321", "I10"],
        },
        {
            "patient_id": "p4", "visit_id": "v4",
            "text": ["heart", "failure", "with", "reduced", "ejection", "fraction", "diuretic", "therapy"],
            "icd_codes": ["I11.0", "I21.09", "K21.0"],
        },
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"text": "sequence"},
        output_schema={"icd_codes": "multilabel"},
        dataset_name="mimic4_icd10_synthetic",
    )


def load_mimic4_dataset(data_dir: str, dev: bool = False):
    """Load MIMIC-IV data and apply the ICD-10 coding task."""
    ds = MIMIC4Dataset(
        ehr_root=data_dir,
        note_root=data_dir,
        ehr_tables=["diagnoses_icd"],
        note_tables=["discharge"],
        dev=dev,
    )
    task = MIMIC4ICD10Coding()
    return ds.set_task(task)


def train_with_curriculum(model, train_loader, depth_epochs, device="cpu") -> float:
    """Train through progressively finer hierarchy depths, returning final loss."""
    final_loss = 0.0
    for depth in sorted(depth_epochs.keys()):
        model.set_depth(depth)
        epochs = depth_epochs[depth]
        print(f"  Depth {depth} ({model.depth_sizes[depth]} codes): training for {epochs} epochs...")

        trainer = Trainer(model=model, device=device, enable_logging=False)
        trainer.train(
            train_dataloader=train_loader,
            epochs=epochs,
            optimizer_class=torch.optim.Adam,
            optimizer_params={"lr": 1e-3},
        )

        model.eval()
        with torch.no_grad():
            batch = next(iter(train_loader))
            ret = model(**{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
            final_loss = ret["loss"].item()
        model.train()
        print(f"    -> Loss at depth {depth}: {final_loss:.4f}")

    return final_loss


def train_flat(model, train_loader, epochs, device="cpu") -> float:
    """Train at the finest depth only (no curriculum), returning final loss."""
    model.set_depth(2)
    print(f"  Flat training at depth 2 ({model.depth_sizes[2]} codes): {epochs} epochs...")

    trainer = Trainer(model=model, device=device, enable_logging=False)
    trainer.train(
        train_dataloader=train_loader,
        epochs=epochs,
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": 1e-3},
    )

    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        ret = model(**{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
        loss = ret["loss"].item()
    model.train()
    print(f"    -> Final loss: {loss:.4f}")
    return loss


def main() -> None:
    parser = argparse.ArgumentParser(description="HiCu ICD-10 coding example")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to MIMIC-IV data directory (hosp/, note/ subdirs). "
                             "If omitted, uses synthetic data.")
    parser.add_argument("--dev", action="store_true",
                        help="Use dev mode (limit to 1000 patients) for faster iteration.")
    parser.add_argument("--epochs-d0", type=int, default=3, help="Epochs at depth 0 (chapters)")
    parser.add_argument("--epochs-d1", type=int, default=5, help="Epochs at depth 1 (categories)")
    parser.add_argument("--epochs-d2", type=int, default=10, help="Epochs at depth 2 (full codes)")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--num-filter-maps", type=int, default=50, help="CNN filter maps")
    parser.add_argument("--embedding-dim", type=int, default=100, help="Word embedding dimension")
    args = parser.parse_args()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # --- Load dataset ---
    if args.data_dir:
        print(f"\nLoading real MIMIC-IV data from {args.data_dir}...")
        dataset = load_mimic4_dataset(args.data_dir, dev=args.dev)
        print(f"Loaded {len(dataset)} samples")
        # Use larger defaults for real data
        if args.num_filter_maps == 50 and args.embedding_dim == 100:
            print("Using default hyperparameters (num_filter_maps=50, embedding_dim=100)")
    else:
        print("\nUsing synthetic dataset (pass --data-dir for real MIMIC-IV data)")
        dataset = create_synthetic_dataset()
        # Override to smaller dims for synthetic data
        args.num_filter_maps = 16
        args.embedding_dim = 32

    train_loader = get_dataloader(dataset, batch_size=args.batch_size, shuffle=True)

    base_kwargs = dict(
        num_filter_maps=args.num_filter_maps,
        embedding_dim=args.embedding_dim,
        kernel_sizes=[3, 5, 9],
    )
    depth_epochs = {0: args.epochs_d0, 1: args.epochs_d1, 2: args.epochs_d2}
    total_flat_epochs = sum(depth_epochs.values())

    results = {}

    #  Experiment 1: Curriculum + ASL (baseline) 
    print("\n=== Experiment 1: Curriculum + ASL (baseline) ===")
    model1 = HiCu(dataset, **base_kwargs)
    print(f"Hierarchy depths: {model1.depth_sizes}")
    results["curriculum+ASL"] = train_with_curriculum(model1, train_loader, depth_epochs, device)

    #  Experiment 2: Flat training + ASL 
    print("\n=== Experiment 2: Flat (no curriculum) + ASL ===")
    model2 = HiCu(dataset, **base_kwargs)
    results["flat+ASL"] = train_flat(model2, train_loader, total_flat_epochs, device)

    #  Experiment 3: Curriculum + BCE (no ASL) 
    print("\n=== Experiment 3: Curriculum + BCE (no ASL) ===")
    model3 = HiCu(dataset, **base_kwargs, asl_gamma_neg=0.0, asl_gamma_pos=0.0, asl_clip=0.0)
    results["curriculum+BCE"] = train_with_curriculum(model3, train_loader, depth_epochs, device)

    #  Experiment 4: Curriculum + ASL + more filters 
    more_filters = args.num_filter_maps * 2
    print(f"\n=== Experiment 4: Curriculum + ASL + more filters ({more_filters}) ===")
    model4 = HiCu(dataset, num_filter_maps=more_filters, embedding_dim=args.embedding_dim, kernel_sizes=[3, 5, 9])
    results[f"curriculum+ASL+filters{more_filters}"] = train_with_curriculum(
        model4, train_loader, depth_epochs, device
    )

    #  Summary 
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Configuration':<35} {'Final Loss':>12}")
    print("-" * 60)
    for config, loss in results.items():
        print(f"{config:<35} {loss:>12.4f}")
    print("=" * 60)
    if not args.data_dir:
        print(
            "\nNote: These results are on synthetic data."
            "\nAbsolute values are not meaningful; the purpose is to demonstrate"
            "\nthat all code paths execute correctly."
        )


if __name__ == "__main__":
    main()
