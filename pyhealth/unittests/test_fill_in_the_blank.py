# ------------------------------------------------------------------------------
# Unit Tests for HurtfulWordsDataset
#
# Names:
#   Zilal Eiz Al Din && Payel Chakraborty
#
# NetIDs:
#   zelalae2  && payelc2
#
# Purpose:
#   This file contains unit tests for verifying the fill_in_the_blank model output
#   - log probability of a maked word that is a male
#   - log probability of a maked word that is a female
#   - Bias Score
#
# Dataset:
#   from pyhealth.datasets import MIMIC3Dataset
#
# Usage:
#   Run `pytest` to automatically discover and run these tests.
# ------------------------------------------------------------------------------


import torch
from pyhealth.models import FillInTheBlank


def test_fill_in_the_blank_forward():
    """Unit test for FillInTheBlank model using a dummy dataset.

    This test verifies:
    - The model runs without error.
    - The output tensor shape is (1, 2), representing logP("he") and logP("she").
    - The log-probabilities are valid (i.e., negative or zero).
    - The computed bias score is a valid float.

    Returns:
        None. Prints the log-probabilities and bias score.
    """

    # Dummy dataset schema with required fields
    class DummyDataset:
        input_schema = {"masked_text": "text"}
        output_schema = {"bias_score": "regression"}

    # Initialize model
    model = FillInTheBlank(dataset=DummyDataset())

    # Define input batch with a single masked sentence
    batch = {
        "masked_text": ["The [MASK] was admitted to PHIHOSPITALPHI."]
    }

    # Run forward pass
    output = model(batch)

    # --- Assertions ---
    assert isinstance(output, torch.Tensor), "Output must be a tensor"
    assert output.shape == (1, 2), "Output shape must be (1, 2)"

    logp_he, logp_she = output[0].tolist()

    # Validate values are valid log-probabilities
    assert logp_he < 0, "logP(he) should be negative"
    assert logp_she < 0, "logP(she) should be negative"

    # Compute bias score
    bias = logp_he - logp_she
    assert isinstance(bias, float), "Bias score must be a float"

    # Print result
    print(f"logP(he): {round(logp_he, 4)}")
    print(f"logP(she): {round(logp_she, 4)}")
    print(f"Bias Score (he - she): {round(bias, 4)}")

