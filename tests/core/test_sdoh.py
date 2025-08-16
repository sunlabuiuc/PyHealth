from typing import Set
from base import BaseTestCase
from pyhealth.models.sdoh import SdohClassifier


class TestSdoh(BaseTestCase):
    def setUp(self):
        self.set_random_seed()

    def test_sdoh(self):
        # example sentence
        sent = 'Pt is homeless and has no car and has no parents or support'
        sdoh = SdohClassifier()
        preds: Set[str] = sdoh.predict(sent)
        self.assertEqual({'housing', 'transportation'}, preds)
