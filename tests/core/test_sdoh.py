from typing import Set
from tests.base import BaseTestCase
from pyhealth.models.sdoh import SdohClassifier


class TestSdoh(BaseTestCase):
    def setUp(self):
        self.set_random_seed()

    def test_is_initialized_nn_module(self):
        """SdohClassifier must initialize its nn.Module state.

        Regression test: when the class was a @dataclass, the generated
        __init__ skipped nn.Module.__init__, so the module had no _parameters
        or _modules and could not be used as a torch model.
        """
        sdoh = SdohClassifier()
        # These all require nn.Module.__init__ to have run.
        self.assertEqual(len(list(sdoh.parameters())), 1)
        sdoh.eval()
        sdoh.to("cpu")
        self.assertIsNone(sdoh.api_key)
        self.assertEqual(sdoh.base_model_id, "meta-llama/Llama-3.1-8B-Instruct")

    def test_parse_reponse(self):
        """Test the parsing of the SDOH output."""
        import pandas as pd
        df = pd.read_csv('test-resources/core/sdoh-test-cases.csv')
        sdoh = SdohClassifier()
        response: str
        preds: str
        for response, preds in df.itertuples(index=False, name=None):
            parsed: str = ','.join(sorted(sdoh._parse_response(response)))
            parsed = '-' if len(parsed) == 0 else parsed
            self.assertEqual(parsed, preds)

    def test_predict(self):
        """Test SDOH prediction."""
        # example sentence
        sent = 'Pt is homeless and has no car and has no parents or support'
        sdoh = SdohClassifier()
        try:
            preds: Set[str] = sdoh.predict(sent)
            self.assertEqual({'housing', 'transportation'}, preds)
        except OSError as e:
            # but allow pass of the test if downloading the model on GitHub
            # workflows fails to download the model
            if str(e).find('You are trying to access a gated repo') > -1:
                pass
            else:
                raise e
