"""
Unit tests for the BHCToAVS model.

These tests validate both the unit-level behavior of the predict method
(using a mocked pipeline) and an optional integration path that runs
against the real Hugging Face model when credentials are provided.
"""

import os
import unittest
from unittest.mock import patch

from tests.base import BaseTestCase
from pyhealth.models.bhc_to_avs import BHCToAVS


class _DummyPipeline:
    """
    Lightweight mock pipeline used to simulate Hugging Face text generation.

    This avoids downloading models or requiring authentication during unit tests.
    """

    def __call__(self, prompt, **kwargs):
        """Return a fixed, deterministic generated response."""
        return [
            {
                "generated_text": "Your pain improved with supportive care and you were discharged in good condition."
            }
        ]


class TestBHCToAVS(BaseTestCase):
    """Unit and integration tests for the BHCToAVS model."""

    def setUp(self):
        """Set a deterministic random seed before each test."""
        self.set_random_seed()

    def test_predict_unit(self):
        """
        Test the predict method using a mocked pipeline.

        This test verifies that:
        - The model returns a string output
        - The output is non-empty
        - The output differs from the input text
        """
        
        bhc_text = (
            "Patient admitted with abdominal pain. Imaging showed no acute findings. "
            "Pain improved with supportive care and the patient was discharged in stable condition."
        )

        with patch.object(BHCToAVS, "_get_pipeline", return_value=_DummyPipeline()):
            model = BHCToAVS()
            summary = model.predict(bhc_text)

        # Output must be type str
        self.assertIsInstance(summary, str)

        # Output should not be empty
        self.assertGreater(len(summary.strip()), 0)

        # Output should be different from input
        self.assertNotIn(bhc_text[:40], summary)

    @unittest.skipUnless(
        os.getenv("RUN_BHC_TO_AVS_INTEGRATION") == "1" and os.getenv("HF_TOKEN"),
        "Integration test disabled. Set RUN_BHC_TO_AVS_INTEGRATION=1 and HF_TOKEN to enable.",
    )
    def test_predict_integration(self):
        """
        Integration test for the BHCToAVS model.

        This test runs the full inference pipeline using the real Hugging Face model.
        It requires the HF_TOKEN environment variable to be set and is skipped by default.
        """

        # For Mistral weights, you will need HF_TOKEN set in the environment.
        bhc_text = (
            "Patient admitted with abdominal pain. Imaging showed no acute findings. "
            "Pain improved with supportive care and the patient was discharged in stable condition."
        )

        model = BHCToAVS()
        summary = model.predict(bhc_text)

        # Output must be type str
        self.assertIsInstance(summary, str)

        # Output should not be empty
        self.assertGreater(len(summary.strip()), 0)

        # Output should be different from input
        self.assertNotIn(bhc_text[:40], summary)
