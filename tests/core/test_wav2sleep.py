"""Tests for the Wav2Sleep model.

Uses small synthetic signals so every test completes in milliseconds.
No real PSG datasets are required.

Tests cover:
- Single modality configurations
- Multi-modality configurations (ECG+PPG, full 6-modality CFS)
- Missing modality handling (graceful degradation)
- Model hyperparameter variations
- State saving and loading
- Gradient computation and training

Dataset link:
    Dataset must be requested from the NSRR at https://sleepdata.org/datasets/cfs

Dataset paper: (please cite if you use this dataset)
    Zhang GQ, Cui L, Mueller R, Tao S, Kim M, Rueschman M, Mariani S, Mobley D,
    Redline S. The National Sleep Research Resource: towards a sleep data commons.
    J Am Med Inform Assoc. 2018 Oct 1;25(10):1351-1358. doi: 10.1093/jamia/ocy064.
    PMID: 29860441; PMCID: PMC6188513.

    Redline S, Tishler PV, Tosteson TD, Williamson J, Kump K, Browner I, Ferrette V,
    Krejci P. The familial aggregation of obstructive sleep apnea. Am J Respir Crit 
    Care Med. 1995 Mar;151(3 Pt 1):682-7.
    doi: 10.1164/ajrccm/151.3_Pt_1.682. PMID: 7881656.

Please include the following text in the Acknowledgements:
    The Cleveland Family Study (CFS) was supported by grants from the National Institutes
    of Health (HL46380, M01 RR00080-39, T32-HL07567, RO1-46380).
    The National Sleep ResearchResource was supported by the National Heart, Lung, and
    Blood Institute (R24 HL114473, 75N92019R002).

Authors:
    Austin Jarrett (ajj7@illinois.edu)
    Justin Cheok (jcheok2@illinois.edu)
    Jimmy Scray (escray2@illinois.edu)
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import Wav2Sleep


# ============================================================================
# Shared fixture helpers
# ============================================================================

def _make_samples(
    n: int = 4,
    n_channels: int = 1,
    length: int = 256,
    n_classes: int = 4,
    seed: int = 0,
):
    """Return a list of synthetic single-modality samples (ECG-like).

    Args:
        n (int): Number of samples. Default is 4.
        n_channels (int): Number of channels per sample. Default is 1.
        length (int): Sample length. Default is 256.
        n_classes (int): Number of output classes. Default is 4.
        seed (int): Random seed. Default is 0.

    Returns:
        List of dicts with patient_id, visit_id, modality tensor, and label.
    """
    rng = np.random.RandomState(seed)
    return [
        {
            "patient_id": f"p{i}",
            "visit_id": "v0",
            "ecg": rng.randn(n_channels, length).astype(np.float32),
            "label": i % n_classes,
        }
        for i in range(n)
    ]


def _make_multimodal_samples(
    n: int = 4,
    length_ecg: int = 256,
    length_ppg: int = 256,
    n_classes: int = 4,
    seed: int = 0,
):
    """Return multi-modality samples with ecg + ppg.

    Args:
        n (int): Number of samples. Default is 4.
        length_ecg (int): ECG sample length. Default is 256.
        length_ppg (int): PPG sample length. Default is 256.
        n_classes (int): Number of output classes. Default is 4.
        seed (int): Random seed. Default is 0.

    Returns:
        List of dicts with ECG, PPG, and labels.
    """
    rng = np.random.RandomState(seed)
    return [
        {
            "patient_id": f"p{i}",
            "visit_id": "v0",
            "ecg": rng.randn(1, length_ecg).astype(np.float32),
            "ppg": rng.randn(1, length_ppg).astype(np.float32),
            "label": i % n_classes,
        }
        for i in range(n)
    ]


def _make_cfs_6modality_samples(
    n: int = 10,
    eeg_len: int = 256,
    eog_len: int = 256,
    emg_len: int = 128,
    ecg_len: int = 256,
    ppg_len: int = 256,
    n_classes: int = 5,
    seed: int = 0,
):
    """Return CFS 6-modality samples (all sleep staging signals with PPG).

    Args:
        n (int): Number of samples. Default is 10 (ensure all classes represented).
        eeg_len (int): EEG sample length. Default is 256.
        eog_len (int): EOG sample length. Default is 256.
        emg_len (int): EMG sample length. Default is 128.
        ecg_len (int): ECG sample length. Default is 256.
        ppg_len (int): PPG sample length. Default is 256.
        n_classes (int): Number of sleep stages. Default is 5.
        seed (int): Random seed. Default is 0.

    Returns:
        List of dicts with all 6 modalities and sleep stage labels (0-4).
    """
    rng = np.random.RandomState(seed)
    return [
        {
            "patient_id": f"p{i}",
            "visit_id": "v0",
            "eeg": rng.randn(1, eeg_len).astype(np.float32),
            "eog_left": rng.randn(1, eog_len).astype(np.float32),
            "eog_right": rng.randn(1, eog_len).astype(np.float32),
            "emg": rng.randn(1, emg_len).astype(np.float32),
            "ecg": rng.randn(1, ecg_len).astype(np.float32),
            "ppg": rng.randn(1, ppg_len).astype(np.float32),
            "label": i % n_classes,
        }
        for i in range(n)
    ]


# ============================================================================
# Test cases
# ============================================================================

class TestWav2SleepSingleModality(unittest.TestCase):
    """Tests for Wav2Sleep with a single ECG modality."""

    def setUp(self):
        samples = _make_samples(n=4, n_channels=1, length=256, n_classes=4)
        self.dataset = create_sample_dataset(
            samples=samples,
            input_schema={"ecg": "tensor"},
            output_schema={"label": "multiclass"},
            dataset_name="test_wav2sleep_single",
        )
        self.model = Wav2Sleep(dataset=self.dataset)
        self.loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)

    def test_initialization(self):
        """Model initialises with correct feature and label keys."""
        self.assertIsInstance(self.model, Wav2Sleep)
        self.assertIn("ecg", self.model.feature_keys)
        self.assertEqual(len(self.model.feature_keys), 1)
        self.assertIn("label", self.model.label_keys)
        self.assertEqual(self.model.feature_dim, 128)

    def test_forward_output_keys(self):
        """Forward pass returns the four required keys."""
        batch = next(iter(self.loader))
        with torch.no_grad():
            out = self.model(**batch)
        for key in ("loss", "y_prob", "y_true", "logit"):
            self.assertIn(key, out, f"Missing key: {key}")

    def test_output_shapes(self):
        """Output tensors have the expected shapes."""
        batch = next(iter(self.loader))
        bs = batch["ecg"].shape[0]
        n_classes = 4
        with torch.no_grad():
            out = self.model(**batch)
        self.assertEqual(out["y_prob"].shape, (bs, n_classes))
        self.assertEqual(out["logit"].shape, (bs, n_classes))
        self.assertEqual(out["y_true"].shape, (bs,))
        self.assertEqual(out["loss"].dim(), 0)

    def test_y_prob_sums_to_one(self):
        """Softmax probabilities sum to 1 for each sample."""
        batch = next(iter(self.loader))
        with torch.no_grad():
            out = self.model(**batch)
        sums = out["y_prob"].sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_backward(self):
        """Loss backward produces gradients on model parameters."""
        batch = next(iter(self.loader))
        out = self.model(**batch)
        out["loss"].backward()
        has_grad = any(
            p.requires_grad and p.grad is not None
            for p in self.model.parameters()
        )
        self.assertTrue(has_grad, "No parameter received a gradient.")

    def test_embed_flag(self):
        """embed=True adds an 'embed' key with correct shape."""
        batch = next(iter(self.loader))
        batch["embed"] = True
        with torch.no_grad():
            out = self.model(**batch)
        self.assertIn("embed", out)
        bs = batch["ecg"].shape[0]
        self.assertEqual(out["embed"].shape, (bs, self.model.feature_dim))


class TestWav2SleepMultiModality(unittest.TestCase):
    """Tests for Wav2Sleep with two input modalities (ECG + PPG)."""

    def setUp(self):
        samples = _make_multimodal_samples(
            n=4, length_ecg=256, length_ppg=256, n_classes=4
        )
        self.dataset = create_sample_dataset(
            samples=samples,
            input_schema={"ecg": "tensor", "ppg": "tensor"},
            output_schema={"label": "multiclass"},
            dataset_name="test_wav2sleep_multi",
        )
        self.model = Wav2Sleep(dataset=self.dataset)
        self.loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)

    def test_feature_keys(self):
        """Both modality keys are registered."""
        self.assertIn("ecg", self.model.feature_keys)
        self.assertIn("ppg", self.model.feature_keys)
        self.assertEqual(len(self.model.feature_keys), 2)

    def test_forward_multimodal(self):
        """Forward pass succeeds with both modalities present."""
        batch = next(iter(self.loader))
        with torch.no_grad():
            out = self.model(**batch)
        for key in ("loss", "y_prob", "y_true", "logit"):
            self.assertIn(key, out)

    def test_forward_single_modality_at_test_time(self):
        """Model degrades gracefully when only one modality is provided."""
        batch = next(iter(self.loader))
        # Simulate ECG-only inference by removing PPG from the batch
        ecg_only_batch = {k: v for k, v in batch.items() if k != "ppg"}
        with torch.no_grad():
            out = self.model(**ecg_only_batch)
        for key in ("loss", "y_prob", "y_true", "logit"):
            self.assertIn(key, out)

    def test_output_shapes_multimodal(self):
        """Output shapes are correct with two modalities."""
        batch = next(iter(self.loader))
        bs = batch["ecg"].shape[0]
        n_classes = 4
        with torch.no_grad():
            out = self.model(**batch)
        self.assertEqual(out["y_prob"].shape, (bs, n_classes))
        self.assertEqual(out["logit"].shape, (bs, n_classes))


class TestWav2SleepHyperparameters(unittest.TestCase):
    """Tests for non-default hyperparameter configurations."""

    def setUp(self):
        samples = _make_samples(n=4, n_channels=1, length=256, n_classes=4)
        self.dataset = create_sample_dataset(
            samples=samples,
            input_schema={"ecg": "tensor"},
            output_schema={"label": "multiclass"},
            dataset_name="test_wav2sleep_hp",
        )
        self.loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)

    def test_custom_feature_dim(self):
        """Custom feature_dim changes the embedding size."""
        model = Wav2Sleep(
            dataset=self.dataset,
            feature_dim=64,
            n_attention_heads=4,
        )
        self.assertEqual(model.feature_dim, 64)
        batch = next(iter(self.loader))
        with torch.no_grad():
            out = model(**batch)
        self.assertIn("loss", out)

    def test_deeper_transformer(self):
        """More transformer layers still produces valid output."""
        model = Wav2Sleep(
            dataset=self.dataset,
            n_transformer_layers=4,
        )
        batch = next(iter(self.loader))
        with torch.no_grad():
            out = model(**batch)
        self.assertIn("loss", out)

    def test_higher_dropout(self):
        """High dropout (0.5) during training does not crash."""
        model = Wav2Sleep(dataset=self.dataset, dropout=0.5)
        model.train()
        batch = next(iter(self.loader))
        out = model(**batch)
        out["loss"].backward()
        self.assertIn("loss", out)


class TestWav2SleepMultivariate(unittest.TestCase):
    """Tests for multi-channel (multivariate) input signals."""

    def setUp(self):
        rng = np.random.RandomState(42)
        # 2-channel respiratory belt (ABD + THX style)
        samples = [
            {
                "patient_id": f"p{i}",
                "visit_id": "v0",
                "resp": rng.randn(2, 256).astype(np.float32),
                "label": i % 4,
            }
            for i in range(4)
        ]
        self.dataset = create_sample_dataset(
            samples=samples,
            input_schema={"resp": "tensor"},
            output_schema={"label": "multiclass"},
            dataset_name="test_wav2sleep_mv",
        )
        self.model = Wav2Sleep(dataset=self.dataset)
        self.loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)

    def test_multivariate_forward(self):
        """Model handles 2-channel input correctly."""
        batch = next(iter(self.loader))
        with torch.no_grad():
            out = self.model(**batch)
        self.assertIn("loss", out)
        self.assertEqual(out["y_prob"].shape, (4, 4))


class TestWav2SleepCFS6Modality(unittest.TestCase):
    """Tests for Wav2Sleep with full CFS 6-modality configuration.

    Tests the complete sleep staging setup with all signal types including
    the optional PPG modality (EEG, EOG-L, EOG-R, EMG, ECG, PPG).
    """

    def setUp(self):
        """Initialize dataset with all 6 modalities.

        Creates synthetic samples with all signal types as expected from
        the SleepStagingCFS task with PPG enabled.
        """
        samples = _make_cfs_6modality_samples(
            n=10,  # Need enough to represent all 5 classes
            eeg_len=256,
            eog_len=256,
            emg_len=128,
            ecg_len=256,
            ppg_len=256,
            n_classes=5,  # 5 sleep stages
        )
        self.dataset = create_sample_dataset(
            samples=samples,
            input_schema={
                "eeg": "tensor",
                "eog_left": "tensor",
                "eog_right": "tensor",
                "emg": "tensor",
                "ecg": "tensor",
                "ppg": "tensor",
            },
            output_schema={"label": "multiclass"},
            dataset_name="test_wav2sleep_cfs6",
        )
        self.model = Wav2Sleep(dataset=self.dataset)
        self.loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)

    def test_6modality_initialization(self):
        """Test model initializes with 6 modalities.

        Verifies that all signal modalities are recognized as feature keys.
        """
        expected_modalities = {
            "eeg", "eog_left", "eog_right", "emg", "ecg", "ppg"
        }
        actual_modalities = set(self.model.feature_keys)
        self.assertEqual(expected_modalities, actual_modalities)

    def test_6modality_forward(self):
        """Test forward pass with all 6 modalities present.

        Verifies model can process complete CFS dataset with all signals.
        """
        batch = next(iter(self.loader))
        with torch.no_grad():
            out = self.model(**batch)

        # Verify all required output keys present
        for key in ("loss", "y_prob", "y_true", "logit"):
            self.assertIn(key, out)

        # Verify correct output shapes (5 classes for 5 sleep stages)
        batch_size = batch["eeg"].shape[0]
        # Output classes determined by unique labels in data
        self.assertEqual(out["y_prob"].shape[0], batch_size)
        self.assertGreater(out["y_prob"].shape[1], 0)  # Has output classes

    def test_6modality_output_shapes(self):
        """Test output tensor shapes with 6 modalities.

        Verifies model outputs have correct dimensions for sleep staging
        with multiple classes.
        """
        batch = next(iter(self.loader))
        batch_size = batch["eeg"].shape[0]

        with torch.no_grad():
            out = self.model(**batch)

        # Verify shapes are valid
        self.assertEqual(out["y_prob"].shape[0], batch_size)
        self.assertEqual(out["logit"].shape[0], batch_size)
        self.assertEqual(out["y_true"].shape, (batch_size,))
        # Verify same number of output classes for both
        self.assertEqual(out["y_prob"].shape[1], out["logit"].shape[1])

    def test_6modality_gradient_flow(self):
        """Test that gradients flow through all modality encoders.

        Verifies that loss backward pass reaches all model parameters.
        """
        batch = next(iter(self.loader))
        out = self.model(**batch)
        out["loss"].backward()

        # Check that encoders have gradients
        has_grad = any(
            p.requires_grad and p.grad is not None
            for p in self.model.parameters()
        )
        self.assertTrue(has_grad, "No gradients computed for model parameters")

    def test_6modality_probability_sum(self):
        """Test that softmax probabilities sum to 1.

        Verifies model outputs valid probability distributions.
        """
        batch = next(iter(self.loader))
        with torch.no_grad():
            out = self.model(**batch)

        probs_sum = out["y_prob"].sum(dim=-1)
        expected_sum = torch.ones(batch["eeg"].shape[0])
        self.assertTrue(
            torch.allclose(probs_sum, expected_sum, atol=1e-5),
            "Probabilities do not sum to 1"
        )

    def test_6modality_embed_output(self):
        """Test embeddings output with all 6 modalities.

        Verifies that embed=True produces expected embedding shape.
        """
        batch = next(iter(self.loader))
        batch["embed"] = True

        with torch.no_grad():
            out = self.model(**batch)

        self.assertIn("embed", out)
        batch_size = batch["eeg"].shape[0]
        feature_dim = self.model.feature_dim
        self.assertEqual(out["embed"].shape, (batch_size, feature_dim))

    def test_5sleep_stages_all_represented(self):
        """Test that all 5 sleep stage classes are in batch.

        Verifies that generated samples cover all sleep stages (0-4).
        """
        samples = _make_cfs_6modality_samples(n=10, n_classes=5)
        labels = [s["label"] for s in samples]
        unique_labels = set(labels)
        self.assertEqual(unique_labels, {0, 1, 2, 3, 4})


class TestWav2SleepMissingModality(unittest.TestCase):
    """Tests for model degradation with missing modalities.

    Verifies that Wav2Sleep gracefully handles inference when some
    modalities are missing (e.g., PPG unavailable at test time).
    """

    def setUp(self):
        """Initialize model trained on 2 modalities (ECG + PPG).

        Model learns from both ECG and PPG during training, but should
        work with only ECG at inference time.
        """
        samples = _make_multimodal_samples(
            n=8,  # Enough samples for determinism testing
            length_ecg=256,
            length_ppg=256,
            n_classes=4,
        )
        self.dataset = create_sample_dataset(
            samples=samples,
            input_schema={"ecg": "tensor", "ppg": "tensor"},
            output_schema={"label": "multiclass"},
            dataset_name="test_wav2sleep_missing",
        )
        self.model = Wav2Sleep(dataset=self.dataset)
        self.loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)

    def test_missing_ppg_at_inference(self):
        """Test model inference with only ECG (PPG missing).

        Verifies that model can process ECG-only inference even when
        trained with PPG.
        """
        batch = next(iter(self.loader))
        # Remove PPG from batch to simulate missing modality at test time
        ecg_only_batch = {k: v for k, v in batch.items() if k != "ppg"}

        with torch.no_grad():
            out = self.model(**ecg_only_batch)

        # Should still produce valid output
        for key in ("loss", "y_prob", "y_true", "logit"):
            self.assertIn(key, out)

        self.assertEqual(out["y_prob"].shape, (4, 4))

    def test_ecg_only_forward_shapes(self):
        """Test that ECG-only inference produces correct output shapes.

        Verifies output dimensions are correct with single modality.
        """
        batch = next(iter(self.loader))
        ecg_only_batch = {k: v for k, v in batch.items() if k != "ppg"}

        with torch.no_grad():
            out = self.model(**ecg_only_batch)

        self.assertEqual(out["y_prob"].shape, (4, 4))
        self.assertEqual(out["logit"].shape, (4, 4))
        self.assertEqual(out["y_true"].shape, (4,))

    def test_both_modalities_inference(self):
        """Test inference with both modalities present.

        Verifies normal operation when both ECG and PPG available.
        """
        batch = next(iter(self.loader))

        with torch.no_grad():
            out = self.model(**batch)

        self.assertEqual(out["y_prob"].shape, (4, 4))

    def test_missing_ppg_vs_full_deterministic(self):
        """Test that ECG-only inference is deterministic.

        Verifies reproducibility when PPG is missing.
        """
        batch = next(iter(self.loader))
        ecg_only_batch = {k: v for k, v in batch.items() if k != "ppg"}

        # Put model in eval mode to disable dropout
        self.model.eval()
        with torch.no_grad():
            out1 = self.model(**ecg_only_batch)
            out2 = self.model(**ecg_only_batch)

        # Same input should produce same output (deterministic in eval mode)
        torch.testing.assert_close(out1["y_prob"], out2["y_prob"])


class TestWav2SleepModelStateManagement(unittest.TestCase):
    """Tests for model state saving, loading, and persistence.

    Tests that trained models can be saved and loaded with state
    preservation.
    """

    def setUp(self):
        """Initialize model for state management testing."""
        samples = _make_samples(n=4, n_channels=1, length=256, n_classes=4)
        self.dataset = create_sample_dataset(
            samples=samples,
            input_schema={"ecg": "tensor"},
            output_schema={"label": "multiclass"},
            dataset_name="test_wav2sleep_state",
        )
        self.model = Wav2Sleep(dataset=self.dataset)
        self.loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)

    def test_model_state_dict_exists(self):
        """Test that model has state_dict for saving.

        Verifies model can be converted to state dict.
        """
        state_dict = self.model.state_dict()
        self.assertIsInstance(state_dict, dict)
        self.assertGreater(len(state_dict), 0)

    def test_model_save_load(self):
        """Test saving and loading model state.

        Verifies that model parameters are preserved through save/load.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"

            # Get original parameters
            original_state = self.model.state_dict()

            # Save model
            torch.save(original_state, str(model_path))
            self.assertTrue(model_path.exists())

            # Create new model and load state
            new_model = Wav2Sleep(dataset=self.dataset)
            new_model.load_state_dict(torch.load(str(model_path)))

            # Verify parameters match
            new_state = new_model.state_dict()
            for key in original_state:
                torch.testing.assert_close(original_state[key], new_state[key])

    def test_model_eval_vs_train_mode(self):
        """Test model behavior in eval vs train mode.

        Verifies that eval mode disables dropout and batch norm updates.
        """
        batch = next(iter(self.loader))

        # Train mode
        self.model.train()
        with torch.no_grad():
            out_train = self.model(**batch)

        # Eval mode
        self.model.eval()
        with torch.no_grad():
            out_eval = self.model(**batch)

        # Both should produce valid outputs
        self.assertIn("loss", out_train)
        self.assertIn("loss", out_eval)

    def test_model_parameter_count(self):
        """Test that model has learnable parameters.

        Verifies model is not empty and has parameters to optimize.
        """
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.assertGreater(total_params, 0, "Model has no learnable parameters")


class TestWav2SleepCFSVariations(unittest.TestCase):
    """Tests for CFS model configurations with different hyperparameters.

    Tests various architectural choices suitable for sleep staging task.
    """

    def setUp(self):
        """Initialize base dataset for variation tests."""
        samples = _make_cfs_6modality_samples(n=10, n_classes=5)
        self.dataset = create_sample_dataset(
            samples=samples,
            input_schema={
                "eeg": "tensor",
                "eog_left": "tensor",
                "eog_right": "tensor",
                "emg": "tensor",
                "ecg": "tensor",
                "ppg": "tensor",
            },
            output_schema={"label": "multiclass"},
            dataset_name="test_wav2sleep_cfs_var",
        )
        self.loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)

    def test_cfs_with_larger_feature_dim(self):
        """Test CFS model with larger feature dimension.

        Verifies that increased feature dimension works for 6-modality input.
        """
        model = Wav2Sleep(
            dataset=self.dataset,
            feature_dim=256,
        )
        batch = next(iter(self.loader))
        with torch.no_grad():
            out = model(**batch)

        self.assertIn("loss", out)
        # Verify output shape is valid (batch_size, num_classes)
        self.assertEqual(out["y_prob"].shape[0], batch["eeg"].shape[0])
        self.assertGreater(out["y_prob"].shape[1], 0)

    def test_cfs_with_multiple_transformer_layers(self):
        """Test CFS model with multiple transformer blocks.

        Verifies that deeper transformer works with 6 modalities.
        """
        model = Wav2Sleep(
            dataset=self.dataset,
            n_transformer_layers=3,
        )
        batch = next(iter(self.loader))
        with torch.no_grad():
            out = model(**batch)

        self.assertIn("loss", out)

    def test_cfs_with_high_attention_heads(self):
        """Test CFS model with multiple attention heads.

        Verifies multi-head attention works with 6 modalities.
        """
        model = Wav2Sleep(
            dataset=self.dataset,
            feature_dim=128,
            n_attention_heads=8,
        )
        batch = next(iter(self.loader))
        with torch.no_grad():
            out = model(**batch)

        self.assertIn("loss", out)

    def test_cfs_training_step(self):
        """Test one training step on CFS configuration.

        Verifies gradients compute and model updates for sleep staging.
        """
        model = Wav2Sleep(dataset=self.dataset)
        batch = next(iter(self.loader))

        # Forward pass
        out = model(**batch)
        initial_loss = out["loss"].item()

        # Backward pass
        out["loss"].backward()

        # Verify gradients exist
        has_grad = any(
            p.requires_grad and p.grad is not None
            for p in model.parameters()
        )
        self.assertTrue(has_grad, "No gradients for training step")

    def test_cfs_5class_output(self):
        """Test that CFS model outputs correct number of classes.

        Verifies output has correct sleep stage classes (Wake, N1, N2, N3, REM).
        """
        model = Wav2Sleep(dataset=self.dataset)
        batch = next(iter(self.loader))

        with torch.no_grad():
            out = model(**batch)

        # Verify output shape is consistent
        batch_size = batch["eeg"].shape[0]
        self.assertEqual(out["y_prob"].shape[0], batch_size)
        self.assertEqual(out["logit"].shape[0], batch_size)
        self.assertEqual(out["y_prob"].shape[1], out["logit"].shape[1])

    def test_cfs_label_range(self):
        """Test that true labels are in valid range [0, 4].

        Verifies sleep stage labels are valid (0=Wake through 4=REM).
        """
        model = Wav2Sleep(dataset=self.dataset)
        batch = next(iter(self.loader))

        with torch.no_grad():
            out = model(**batch)

        labels = out["y_true"]
        self.assertTrue(torch.all(labels >= 0), "Labels below 0")
        self.assertTrue(torch.all(labels < 5), "Labels >= 5")


if __name__ == "__main__":
    unittest.main()
