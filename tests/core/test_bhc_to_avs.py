from tests.base import BaseTestCase
from pyhealth.models.bhc_to_avs import BHCToAVS


class TestBHCToAVS(BaseTestCase):
    """Unit tests for the BHCToAVS model."""

    def setUp(self):
        self.set_random_seed()

    def test_predict(self):
        """Test the predict method of BHCToAVS."""
        bhc_text = (
            "Patient admitted with abdominal pain. Imaging showed no acute findings. "
            "Pain improved with supportive care and the patient was discharged in stable condition."
        )
        model = BHCToAVS()
        try:

            summary = model.predict(bhc_text)

            # Output must be type str
            self.assertIsInstance(summary, str)

            # Output should not be empty
            self.assertGreater(len(summary.strip()), 0)

            # Output should be different from input
            self.assertNotIn(bhc_text[:40], summary)

        except OSError as e:
            # Allow test to pass if model download fails on e.g. on GitHub workflows
            if "gated repo" in str(e).lower() or "404" in str(e):
                pass
            else:
                raise e
