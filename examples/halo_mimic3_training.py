#!/usr/bin/env python3
"""
Train HALO on MIMIC-III dataset.
Example script demonstrating HALO training with configurable parameters.
"""

import os
import argparse
import torch
import pickle
import shutil
from pyhealth.datasets.halo_mimic3 import HALO_MIMIC3Dataset
from pyhealth.models.generators.halo import HALO
from pyhealth.models.generators.halo_resources.halo_config import HALOConfig


def main():
    parser = argparse.ArgumentParser(description="Train HALO on MIMIC-III dataset")
    parser.add_argument("--mimic3_dir", required=True, help="Path to MIMIC-III data directory")
    parser.add_argument("--output_dir", required=True, help="Directory for saving checkpoints and results")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs (default: 80)")
    parser.add_argument("--batch_size", type=int, default=48, help="Training batch size (default: 48)")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate (default: 0.0001)")
    parser.add_argument("--save_best", action="store_true", help="Save best checkpoint (lowest validation loss)")
    parser.add_argument("--save_final", action="store_true", help="Save final checkpoint after training")
    args = parser.parse_args()

    # Setup directories
    pkl_data_dir = os.path.join(args.output_dir, "pkl_data/")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(pkl_data_dir, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

    # Load and preprocess dataset
    print(f"\n{'='*60}", flush=True)
    print("Loading and preprocessing MIMIC-III dataset...", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Data directory: {args.mimic3_dir}", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)

    dataset = HALO_MIMIC3Dataset(
        mimic3_dir=args.mimic3_dir,
        pkl_data_dir=pkl_data_dir,
        gzip=False
    )

    print(f"\n{'='*60}", flush=True)
    print("Dataset preprocessing complete!", flush=True)
    print(f"{'='*60}", flush=True)

    # Load vocabulary sizes
    code_to_index = pickle.load(open(f"{pkl_data_dir}/codeToIndex.pkl", "rb"))
    id_to_label = pickle.load(open(f"{pkl_data_dir}/idToLabel.pkl", "rb"))

    code_vocab_size = len(code_to_index)
    label_vocab_size = len(id_to_label)
    special_vocab_size = 3
    total_vocab_size = code_vocab_size + label_vocab_size + special_vocab_size

    print(f"Vocabulary sizes:", flush=True)
    print(f"  Code vocabulary: {code_vocab_size}", flush=True)
    print(f"  Label vocabulary: {label_vocab_size}", flush=True)
    print(f"  Special tokens: {special_vocab_size}", flush=True)
    print(f"  Total vocabulary: {total_vocab_size}", flush=True)

    # HALO configuration
    print(f"\n{'='*60}", flush=True)
    print("Initializing HALO configuration", flush=True)
    print(f"{'='*60}", flush=True)

    config = HALOConfig(
        total_vocab_size=total_vocab_size,
        code_vocab_size=code_vocab_size,
        label_vocab_size=label_vocab_size,
        special_vocab_size=special_vocab_size,
        n_positions=56,
        n_ctx=48,
        n_embd=768,
        n_layer=12,
        n_head=12,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        batch_size=args.batch_size,
        sample_batch_size=256,
        epoch=args.epochs,
        pos_loss_weight=None,
        lr=args.learning_rate
    )

    print("Configuration:", flush=True)
    print(f"  Embedding dim: {config.n_embd}", flush=True)
    print(f"  Layers: {config.n_layer}", flush=True)
    print(f"  Attention heads: {config.n_head}", flush=True)
    print(f"  Batch size: {config.batch_size}", flush=True)
    print(f"  Epochs: {config.epoch}", flush=True)
    print(f"  Learning rate: {config.lr}", flush=True)

    # Train HALO model
    print(f"\n{'='*60}", flush=True)
    print("Training HALO model...", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Training for {args.epochs} epochs", flush=True)
    print(f"Progress updates every 1,000 iterations", flush=True)
    print(f"Checkpoints saved when validation loss improves", flush=True)
    print(f"{'='*60}\n", flush=True)

    model = HALO(
        dataset=dataset,
        config=config,
        save_dir=args.output_dir,
        train_on_init=True
    )

    print(f"\n{'='*60}", flush=True)
    print("TRAINING COMPLETE!", flush=True)
    print(f"{'='*60}", flush=True)

    # Save final checkpoint if requested
    if args.save_final:
        final_state = {
            'model': model.model.state_dict(),
            'optimizer': model.optimizer.state_dict(),
            'iteration': 'final',
            'epoch': config.epoch
        }
        torch.save(final_state, os.path.join(args.output_dir, 'halo_model_final'))
        print(f"Final checkpoint saved to: {args.output_dir}/halo_model_final", flush=True)

    # Copy best checkpoint if requested
    if args.save_best:
        best_path = os.path.join(args.output_dir, 'halo_model')
        if os.path.exists(best_path):
            shutil.copy(best_path, os.path.join(args.output_dir, 'halo_model_best'))
            print(f"Best checkpoint copied to: {args.output_dir}/halo_model_best", flush=True)

    print(f"Vocabulary files saved to: {pkl_data_dir}", flush=True)
    print(f"\nTraining artifacts:", flush=True)
    print(f"  - Checkpoints: {args.output_dir}", flush=True)
    print(f"  - Vocabulary: {pkl_data_dir}", flush=True)
    print(f"\nNext step: Generate synthetic data using trained checkpoint", flush=True)


if __name__ == "__main__":
    main()
