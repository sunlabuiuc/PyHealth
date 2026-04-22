import io
from contextlib import redirect_stderr, redirect_stdout
from typing import Dict, List

import numpy as np
import torch

from pyhealth.datasets import eICUDataset, get_dataloader
from pyhealth.datasets.splitter import split_by_patient
from pyhealth.models import RNN
from pyhealth.tasks import FutureSeverityPredictionEICU
from pyhealth.trainer import Trainer


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_experiment(future_window: int, seed: int = 42) -> Dict[str, float]:
    """Runs one ablation setting for future severity prediction."""
    set_seed(seed)

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        dataset = eICUDataset(
            root="test-resources/core/eicudemo",
            tables=["diagnosis", "medication", "physicalexam"],
        )

        task = FutureSeverityPredictionEICU(future_window=future_window)
        sample_dataset = dataset.set_task(task)

        if len(sample_dataset) < 3:
            raise ValueError(
                f"Not enough samples for future_window={future_window}. "
                f"Got {len(sample_dataset)} samples."
            )

        train_dataset, val_dataset, test_dataset = split_by_patient(
            sample_dataset,
            ratios=[0.8, 0.1, 0.1],
        )

        model = RNN(dataset=sample_dataset)

        trainer = Trainer(
            model=model,
            metrics=["accuracy", "f1_macro"],
        )

        train_dataloader = get_dataloader(
            train_dataset,
            batch_size=8,
            shuffle=True,
        )
        val_dataloader = get_dataloader(
            val_dataset,
            batch_size=8,
            shuffle=False,
        )
        test_dataloader = get_dataloader(
            test_dataset,
            batch_size=8,
            shuffle=False,
        )

        trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epochs=3,
        )

        results = trainer.evaluate(test_dataloader)

    return results


def ablation_study(future_windows: List[int], seeds: List[int]) -> None:
    """Runs an ablation over future prediction windows and random seeds."""
    all_results = {}

    print("Future severity prediction ablation")
    print("setting=future_window")
    print("-" * 50)

    for window in future_windows:
        window_results = {"accuracy": [], "f1_macro": []}
        print(f"\nwindow={window}")

        for seed in seeds:
            results = run_experiment(future_window=window, seed=seed)
            window_results["accuracy"].append(results["accuracy"])
            window_results["f1_macro"].append(results["f1_macro"])

            print(
                f"  seed={seed} -> "
                f"{{accuracy={results['accuracy']:.4f}, "
                f"f1_macro={results['f1_macro']:.4f}}}"
            )

        avg_acc = np.mean(window_results["accuracy"])
        std_acc = np.std(window_results["accuracy"])
        avg_f1 = np.mean(window_results["f1_macro"])
        std_f1 = np.std(window_results["f1_macro"])

        all_results[window] = {
            "accuracy_mean": avg_acc,
            "accuracy_std": std_acc,
            "f1_mean": avg_f1,
            "f1_std": std_f1,
        }

        print(
            f"  summary -> "
            f"{{accuracy={avg_acc:.4f}±{std_acc:.4f}, "
            f"f1_macro={avg_f1:.4f}±{std_f1:.4f}}}"
        )

    print("\nfinal_results")
    print("-" * 50)
    for window, res in all_results.items():
        print(
            f"  window={window} -> "
            f"{{accuracy={res['accuracy_mean']:.4f}±{res['accuracy_std']:.4f}, "
            f"f1_macro={res['f1_mean']:.4f}±{res['f1_std']:.4f}}}"
        )


if __name__ == "__main__":
    future_windows = [1, 2]
    seeds = [42, 7, 123]
    ablation_study(future_windows=future_windows, seeds=seeds)