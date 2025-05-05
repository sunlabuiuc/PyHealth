# ==============================================================================
# Author(s): Sharim Khan, Gabriel Lee
# NetID(s): sharimk2, gjlee4
# Paper title:
#           Explaining A Machine Learning Decision to Physicians via Counterfactuals
# Paper link: https://arxiv.org/abs/2306.06325
# Description: Test script to train and evaluate a Counterfactual VAE (CFVAE) on
#              MIMIC-IV for mortality prediction using PyHealth, including training
#              a frozen dummy classifier and then CFVAE with that classifier.
# ==============================================================================

import logging
import os
import sys
from typing import Any

import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to sys.path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def test_cfvae_mortality_prediction_mimic4() -> None:
    """Trains a CFVAE model on MIMIC-IV demo data with a frozen dummy classifier.

    Steps:
        - Load and preprocess MIMIC-IV lab data.
        - Train a binary classifier on in-hospital mortality.
        - Freeze the classifier.
        - Train a CFVAE model to produce counterfactuals.
        - Evaluate CFVAE on test data.
    """
    logger.info("===== Starting CFVAE Unit Test =====")
    from pyhealth.datasets import MIMIC4Dataset
    from pyhealth.tasks import InHospitalMortalityMIMIC4
    from pyhealth.datasets import split_by_sample, get_dataloader
    from pyhealth.trainer import Trainer
    from pyhealth.models import BaseModel, CFVAE

    # Load MIMIC-IV demo dataset
    dataset = MIMIC4Dataset(
        ehr_root="https://physionet.org/files/mimic-iv-demo/2.2/",
        ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"],
    )

    task = InHospitalMortalityMIMIC4()
    samples = dataset.set_task(task)
    logger.info(f"===== Loaded {len(samples)} samples. ===== ")

    # Preprocessing: mean over time, normalize across samples
    logger.info("===== Preprocessing samples (mean over time) =====")
    for sample in samples:
        sample["labs"] = torch.mean(sample["labs"], dim=0)

    labs_tensor = torch.stack([s["labs"] for s in samples])
    feature_mean = labs_tensor.mean(dim=0)
    feature_std = labs_tensor.std(dim=0) + 1e-6

    for sample in samples:
        sample["labs"] = (sample["labs"] - feature_mean) / feature_std

    # Split data
    train_dataset, val_dataset, test_dataset = split_by_sample(
        dataset=samples,
        ratios=[0.7, 0.1, 0.2]
    )

    train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    logger.info("===== Stage 1: Train the dummy classifier =====")

    class DummyClassifier(nn.Module):
        """Simple feedforward binary classifier."""

        def __init__(self, input_dim: int = 27, hidden_dim: int = 64):
            """
            Args:
                input_dim: Dimension of input feature vector.
                hidden_dim: Size of hidden layer.
            """
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Tensor of shape [batch_size, input_dim].

            Returns:
                Output logits as a tensor of shape [batch_size, 1].
            """
            return self.model(x)

    class WrappedClassifier(BaseModel):
        """Wraps a PyTorch classifier into the PyHealth BaseModel interface."""

        def __init__(self, dataset: Any, model: nn.Module):
            """
            Args:
                dataset: PyHealth dataset object.
                model: PyTorch model to be wrapped.
            """
            super().__init__(dataset)
            self.model = model
            self.mode = self.dataset.output_schema[self.label_keys[0]]

        def forward(self, **kwargs) -> dict:
            """Forward pass and loss computation.

            Args:
                kwargs: Dict containing "labs" and "mortality".

            Returns:
                Dictionary with keys "loss", "y_prob", and "y_true".
            """
            x = kwargs[self.feature_keys[0]].to(self.device)
            y = kwargs[self.label_keys[0]].to(self.device)
            logits = self.model(x)
            loss = self.get_loss_function()(logits, y)
            y_prob = self.prepare_y_prob(logits)
            return {
                "loss": loss,
                "y_prob": y_prob,
                "y_true": y
            }

    clf = DummyClassifier(input_dim=27)
    wrapped_model = WrappedClassifier(dataset=samples, model=clf)

    trainer = Trainer(model=wrapped_model, metrics=["roc_auc", "accuracy"])
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=5,
        monitor="roc_auc"
    )

    logger.info("===== Freezing the classifier... =====")
    clf.eval()
    for param in clf.parameters():
        param.requires_grad = False

    logger.info("===== Stage 2: Train CFVAE with frozen classifier =====")

    cfvae_model = CFVAE(
        dataset=samples,
        feature_keys=["labs"],
        label_key="mortality",
        mode="binary",
        feat_dim=27,
        latent_dim=32,
        hidden_dim=64,
        external_classifier=clf
    )

    cfvae_trainer = Trainer(model=cfvae_model, metrics=["roc_auc", "accuracy"])
    cfvae_trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=10,
        monitor="roc_auc",
        optimizer_params={"lr": 1e-3}
    )

    logger.info("===== Test set evaluation =====")
    print(cfvae_trainer.evaluate(test_dataloader))
    logger.info("===== Successfully completed CFVAE unit test! =====")


if __name__ == "__main__":
    test_cfvae_mortality_prediction_mimic4()

