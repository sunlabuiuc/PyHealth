from unittest import mock

from base import BaseTestCase
from pyhealth.models.sdoh_icd9_llm import SDOHICD9LLM


class TestSdohLLM(BaseTestCase):
    def setUp(self):
        self.set_random_seed()

    def test_llm_aggregation(self):
        model = SDOHICD9LLM(dry_run=True)
        notes = ["note one", "note two", "note three"]

        responses = iter(
            [
                "```V600```",
                "None",
                "```V620, V625```",
            ]
        )

        with mock.patch.object(model, "_call_openai_api", side_effect=lambda _: next(responses)):
            with mock.patch.object(model, "_write_prompt_preview", return_value=None):
                aggregated, note_results = model.predict_admission_with_notes(notes)

        self.assertEqual({"V600", "V620", "V625"}, aggregated)
        self.assertEqual(3, len(note_results))
        self.assertEqual({"V600"}, set(note_results[0]["predicted_codes"]))
        self.assertEqual(set(), set(note_results[1]["predicted_codes"]))
        self.assertEqual({"V620", "V625"}, set(note_results[2]["predicted_codes"]))

    def test_llm_max_notes(self):
        model = SDOHICD9LLM(dry_run=True, max_notes=1)
        notes = ["note one", "note two"]

        with mock.patch.object(model, "_call_openai_api", return_value="V600") as mocked:
            with mock.patch.object(model, "_write_prompt_preview", return_value=None):
                aggregated, note_results = model.predict_admission_with_notes(notes)

        self.assertEqual({"V600"}, aggregated)
        self.assertEqual(1, len(note_results))
        mocked.assert_called_once()
