"""Tests for GiantMed's retrieval, prompt and probability refinement."""

import numpy as np
import pytest

from pyhealth.models.giantmed import GiantMed


@pytest.fixture
def visits():
    return [
        {
            "patient_id": "train1",
            "visit_id": "v1",
            "conditions": [["a", "b"]],
            "procedures": [["p"]],
            "drugs": ["B", "C"],
        },
        {
            "patient_id": "train2",
            "visit_id": "v2",
            "conditions": [["a"]],
            "procedures": [["q"]],
            "drugs": ["C"],
        },
    ]


def make(visits, response, **kwargs):
    return GiantMed(
        ["A", "B", "C", "D"],
        [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        lambda prompt: response,
        visits,
        **kwargs,
    )


def test_refinement_and_evidence(visits):
    model = make(visits, '{"B": 0.8, "C": 0.1, "A": 0.0}')
    query = {
        "patient_id": "test",
        "conditions": [["old"], ["a", "b"]],
        "procedures": [["z"], ["p"]],
        "drugs_hist": [["D"], []],
        "drugs": ["SECRET_TARGET"],
    }
    original = np.array([0.9, 0.3, 0.69, 0.7])
    refined, prompt = model.refine(query, original)
    np.testing.assert_allclose(refined, [0.9, 0.8, 0.1, 0.7])
    np.testing.assert_allclose(original, [0.9, 0.3, 0.69, 0.7])
    assert "SECRET_TARGET" not in prompt
    assert '"B", "C"' in prompt
    assert '"overlapping_diagnoses": ["a (a)", "b (b)"]' in prompt
    assert 'Previous medications: [["D (D)"]]' in prompt


def test_retrieval_patient_exclusion_and_filter(visits):
    model = make(visits, '{"B": 0.8}', evidence_filter=lambda q, r, c: c != "B")
    query = {"patient_id": "train2", "conditions": [["a"]], "procedures": [[]]}
    assert [r["patient_id"] for r in model.retrieve(query)] == ["train1"]
    _, prompt = model.refine(query, [0.9, 0.5, 0.1, 0.1])
    assert '"medications": []' in prompt


@pytest.mark.parametrize(
    "response", ["garbage", '{"B": 0.9, "B": 0.1}', '{"B": 2}', '{"B": true}', "[1,2]"]
)
def test_bad_responses_keep_original(visits, response):
    model = make(visits, response)
    result, _ = model.refine({"patient_id": "test"}, [0.1, 0.5, 0.9, 0.9])
    np.testing.assert_allclose(result, [0.1, 0.5, 0.9, 0.9])


def test_validation_and_empty_boundary(visits):
    model = make(visits, "{}")
    with pytest.raises(ValueError):
        model.refine({"patient_id": "test"}, [0.1, float("nan"), 0.9, 0.9])
    with pytest.raises(ValueError):
        model.refine({}, [0.1, 0.5, 0.9, 0.9])
    result, prompt = model.refine({}, [0.1, 0.2, 0.8, 0.9])
    assert prompt is None
    np.testing.assert_allclose(result, [0.1, 0.2, 0.8, 0.9])


def test_medi_evidence_uses_training_visit_and_previous_history():
    from pyhealth.models.giantmed import MediEvidenceFilter

    filt = MediEvidenceFilter(
        {("train1", "v1", "B"): ["rx1"]}, {"rx1": ["250.0"]}, ancestor_match=True
    )
    query = {"conditions": [["25001"]], "drugs_hist": [["C"], []], "drugs": ["D"]}
    neighbour = {"patient_id": "train1", "visit_id": "v1"}
    assert filt(query, neighbour, "B")
    assert filt(query, neighbour, "C")
    assert not filt(query, neighbour, "D")
    assert not filt(query, {"patient_id": "test", "visit_id": "v1"}, "B")


def test_response_coverage_and_paper_boundary(visits):
    model = make(visits, '{"B": 0.8, "C": "bad"}', upper_inclusive=True)
    outcome = model.refine_with_details({"patient_id": "test"}, [0.1, 0.3, 0.7, 0.9])
    assert outcome["candidates"] == ["B", "C"]
    assert outcome["updated"] == ["B"]
    assert outcome["status"] == "partial"
    np.testing.assert_allclose(outcome["probabilities"], [0.1, 0.8, 0.7, 0.9])
    assert "[0.3, 0.7]" in outcome["prompt"]
    model.generate = lambda prompt: "bad"
    assert (
        model.refine_with_details({"patient_id": "test"}, [0.1, 0.5, 0.9, 0.9])[
            "status"
        ]
        == "invalid_response"
    )


def test_medi_reference_files_restricts_training_and_profile(tmp_path):
    from pyhealth.models.giantmed import MediEvidenceFilter

    (tmp_path / "prescriptions.csv").write_text(
        "subject_id,hadm_id,ndc\n1,11,123\n2,22,123\n"
    )
    (tmp_path / "medi.csv").write_text(
        "RXCUI,CODE,SAB,MEDI1,MEDI1_HPS,MEDI2_HPS\n"
        "ingredient,250.00,ICD9CM,TRUE,FALSE,FALSE\n"
        "ingredient,4019,ICD9CM,FALSE,FALSE,FALSE\n"
    )
    (tmp_path / "ndc.txt").write_text("{'123': 'clinical'}")
    (tmp_path / "atc.csv").write_text("RXCUI,ATC4\nclinical,A10AB\n")
    (tmp_path / "cache.json").write_text(
        '{"clinical_rxcuis": {"clinical": {"ingredients": [{"rxcui": "ingredient"}]}}}'
    )
    filt = MediEvidenceFilter.from_reference_files(
        [{"patient_id": "1", "visit_id": "11"}],
        tmp_path / "prescriptions.csv",
        tmp_path / "medi.csv",
        tmp_path / "ndc.txt",
        tmp_path / "atc.csv",
        tmp_path / "cache.json",
        profile="medi1",
    )
    neighbour = {"patient_id": "1", "visit_id": "11"}
    assert filt({"conditions": [["25000"]]}, neighbour, "A10A")
    assert not filt({"conditions": [["4019"]]}, neighbour, "A10A")
    assert not filt(
        {"conditions": [["25000"]]}, {"patient_id": "2", "visit_id": "22"}, "A10A"
    )


def test_huggingface_adapter_decodes_only_generated_tokens():
    import torch
    from pyhealth.models.giantmed import HuggingFaceGiantMedGenerator

    class Batch(dict):
        def __init__(self):
            super().__init__(input_ids=torch.tensor([[1, 2]]))
            self.input_ids = self["input_ids"]

        def to(self, device):
            return self

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "prompt"}]
            assert kwargs["enable_thinking"] is False
            return "rendered"

        def __call__(self, chat, return_tensors):
            assert chat == "rendered" and return_tensors == "pt"
            return Batch()

        def batch_decode(self, tokens, skip_special_tokens):
            assert tokens.tolist() == [[3, 4]]
            return ['  {"B": 0.8}  ']

    class Model:
        device = "cpu"

        def generate(self, **kwargs):
            assert kwargs["temperature"] == 0.6
            assert kwargs["max_new_tokens"] == 5
            return torch.tensor([[1, 2, 3, 4]])

    generate = HuggingFaceGiantMedGenerator(Model(), Tokenizer(), max_new_tokens=5)
    assert generate("prompt") == '{"B": 0.8}'


def test_example_metrics_use_half_threshold_and_patient_weights():
    from examples.drug_recommendation.giantmed_mimic3 import evaluate_predictions

    metrics = evaluate_predictions(
        [[1, 0], [1, 0], [1, 0]],
        [[0.4, 0.1], [0.4, 0.1], [0.9, 0.1]],
        ["first", "first", "second"],
        np.zeros((2, 2)),
    )
    assert metrics["jaccard_samples"] == pytest.approx(0.5)
    assert metrics["f1_samples"] == pytest.approx(0.5)
    assert metrics["mean_medications"] == pytest.approx(0.5)


def test_reject_invalid_ddi_matrix(visits):
    with pytest.raises(ValueError, match="binary"):
        GiantMed(["A"], [[float("nan")]], lambda prompt: "{}", visits)


def test_nested_probability_and_changed_reporting(visits):
    model = make(visits, '{"B": {"probability": 0.8}, "C": {"probability": 0.5}}')
    result = model.refine_with_details({"patient_id": "test"}, [0.1, 0.4, 0.5, 0.9])
    assert result["status"] == "complete"
    assert result["updated"] == ["B", "C"]
    assert result["changed"] == ["B"]
    model.generate = lambda prompt: '{"B": {"probability": 0.8, "probability": 0.2}}'
    result = model.refine_with_details({"patient_id": "test"}, [0.1, 0.4, 0.9, 0.9])
    assert result["updated"] == []


def test_medi_rejects_missing_columns(tmp_path):
    from pyhealth.models.giantmed import MediEvidenceFilter

    (tmp_path / "bad.csv").write_text("RXCUI,CODE\nx,y\n")
    with pytest.raises(ValueError, match="missing required columns"):
        MediEvidenceFilter.from_reference_files(
            [], "unused", tmp_path / "bad.csv", "unused", "unused", "unused"
        )


def test_openrouter_example_adapter_uses_environment_and_returns_content(monkeypatch):
    import json
    from examples.drug_recommendation.giantmed_mimic3 import OpenRouterGenerator

    captured = {}

    class Response:
        def __enter__(self):
            import io

            payload = {
                "choices": [{"message": {"content": '{"A": 0.8}'}}],
                "usage": {"prompt_tokens": 42},
            }
            return io.BytesIO(json.dumps(payload).encode())

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setenv("OPENROUTER_API_KEY", "example-secret")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    generator = OpenRouterGenerator("qwen/qwen3-8b", 123)
    assert generator("dummy prompt") == '{"A": 0.8}'
    assert captured["body"]["model"] == "qwen/qwen3-8b"
    assert captured["body"]["messages"] == [{"role": "user", "content": "dummy prompt"}]
    assert captured["body"]["reasoning"] == {"enabled": False}
    assert captured["authorization"] == "Bearer example-secret"
    assert generator.usage == [{"prompt_tokens": 42}]


def test_token_budget_preserves_core_and_reports_evidence_omissions(visits):
    query = {
        "patient_id": "test",
        "conditions": [["old"], ["a", "b"]],
        "procedures": [["z"], ["p"]],
        "drugs_hist": [["D"] * 40, []],
    }
    model = make(visits, '{"B": 0.8, "C": 0.1}')
    unbounded, _ = model.refine(query, [0.9, 0.3, 0.69, 0.8])
    model.prompt_token_counter = len
    model.max_prompt_tokens = 1400
    result = model.refine_with_details(query, [0.9, 0.3, 0.69, 0.8])
    assert len(result["prompt"]) <= 1400
    assert '["B",0.3,' in result["prompt"] and '["C",0.69,' in result["prompt"]
    assert '"a (a)"' in result["prompt"] and '"b (b)"' in result["prompt"]
    assert result["prompt_budget"]["history_visits"] == [0, 1]
    np.testing.assert_allclose(result["probabilities"], unbounded)


def test_budget_rejects_core_before_generation(visits):
    def forbidden(prompt):
        raise AssertionError("generator must not be called")

    model = GiantMed(
        ["A"], [[0]], forbidden, visits, prompt_token_counter=len, max_prompt_tokens=10
    )
    with pytest.raises(ValueError, match="current visit and candidate scores"):
        model.refine_with_details({"patient_id": "test"}, [0.5])


def test_chat_token_counter_counts_ids_not_encoding_keys():
    from examples.drug_recommendation.giantmed_mimic3 import count_chat_tokens

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["add_generation_prompt"]
            return {"input_ids": [1, 2, 3, 4, 5], "attention_mask": [1] * 5}

    assert count_chat_tokens(Tokenizer(), "hello") == 5


def test_openrouter_stops_if_provider_reports_over_limit(monkeypatch):
    import io
    import json
    from examples.drug_recommendation.giantmed_mimic3 import OpenRouterGenerator

    class Response:
        def __enter__(self):
            return io.BytesIO(
                json.dumps(
                    {
                        "usage": {"prompt_tokens": 4097},
                        "choices": [{"message": {"content": "{}"}}],
                    }
                ).encode()
            )

        def __exit__(self, *args):
            return False

    monkeypatch.setenv("OPENROUTER_API_KEY", "example-secret")
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: Response())
    generator = OpenRouterGenerator("qwen/qwen3-14b", 1024, max_input_tokens=4096)
    with pytest.raises(ValueError, match="4097 input tokens"):
        generator("dummy")


def test_budget_reports_truncated_interactions_without_losing_candidates():
    codes = [f"D{i:02}" for i in range(30)]
    model = GiantMed(
        codes,
        np.ones((30, 30)),
        lambda prompt: "{}",
        [],
        prompt_token_counter=len,
        max_prompt_tokens=2500,
    )
    result = model.refine_with_details(
        {"patient_id": "heldout", "conditions": [["x"]], "procedures": [["y"]]},
        [0.5] * 30,
    )
    assert len(result["prompt"]) <= 2500
    assert len(result["candidates"]) == 30
    assert all(f'["{code}",0.5,' in result["prompt"] for code in codes)
    included, total = result["prompt_budget"]["ddi_pairs"]
    assert 0 < included < total == 435
    assert "adjacency list" in result["prompt"]


def test_indexed_prompt_preserves_interactions_names_and_retrieval(visits):
    import json

    names = {"drugs": {"B": "drug B", "C": "drug C"}}
    model = make(
        visits, "{}", prompt_token_counter=len, max_prompt_tokens=5000, code_names=names
    )
    query = {"patient_id": "test", "conditions": [["a", "b"]], "procedures": [["p"]]}
    result = model.refine_with_details(query, [0.9, 0.4, 0.5, 0.9])
    lines = result["prompt"].splitlines()

    def read_section(label):
        return json.loads(
            next(x[len(label) + 2 :] for x in lines if x.startswith(label + ": "))
        )

    candidates = read_section("Candidates")
    assert candidates == [["B", 0.4, "drug B"], ["C", 0.5, "drug C"]]
    links = read_section("Potential interactions among candidates")
    assert {
        (candidates[int(i)][0], candidates[j][0]) for i, js in links.items() for j in js
    } == {("B", "C")}
    neighbours = read_section("Similar training visits")
    assert neighbours[0]["diagnoses"] == ["a", "b"]
    assert [candidates[i][0] for i in neighbours[0]["medications"]] == ["B", "C"]
    assert result["prompt_budget"]["candidate_names"] == [2, 2]
    assert result["prompt_budget"]["ddi_pairs"] == [1, 1]


def test_batches_preserve_cross_batch_interactions_and_only_update_targets():
    import json
    import re

    codes = ["A", "B", "C", "D"]
    ddi = np.zeros((4, 4))
    for a, b in [(0, 1), (0, 2), (1, 3), (2, 3)]:
        ddi[a, b] = ddi[b, a] = 1
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        target_count = int(re.search(r"Only the first (\d+) rows", prompt).group(1))
        rows = json.loads(
            next(
                x.removeprefix("Candidates: ")
                for x in prompt.splitlines()
                if x.startswith("Candidates: ")
            )
        )
        # Incorrect unsolicited updates to context drugs must be ignored.
        return json.dumps(
            {row[0]: (0.8 if i < target_count else 0.01) for i, row in enumerate(rows)}
        )

    model = GiantMed(
        codes,
        ddi,
        generate,
        [],
        code_names={"drugs": {c: f"name-{c}" for c in codes}},
        prompt_token_counter=len,
        max_prompt_tokens=3000,
        candidate_batch_size=2,
    )
    output = model.refine_with_details({"patient_id": "heldout"}, [0.4] * 4)
    np.testing.assert_allclose(output["probabilities"], [0.8] * 4)
    assert output["status"] == "complete"
    budget = output["prompt_budget"]
    assert budget["batch_count"] == 2
    assert budget["candidate_names"] == [4, 4]
    assert budget["resolved_candidate_names"] == [4, 4]
    assert budget["ddi_pairs"] == [4, 4]
    assert budget["cross_batch_pairs"] == 2
    expected = {("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")}
    seen = set()
    for prompt in prompts:
        assert len(prompt) <= 3000
        rows = json.loads(
            next(
                x.removeprefix("Candidates: ")
                for x in prompt.splitlines()
                if x.startswith("Candidates: ")
            )
        )
        adjacency = json.loads(
            next(
                x.removeprefix("Potential interactions: ")
                for x in prompt.splitlines()
                if x.startswith("Potential interactions: ")
            )
        )
        seen.update(
            tuple(sorted((rows[int(i)][0], rows[j][0])))
            for i, js in adjacency.items()
            for j in js
        )
    assert seen == expected


def test_batch_planning_fails_before_any_generation_if_single_target_does_not_fit():
    calls = []
    model = GiantMed(
        ["A"],
        [[0]],
        lambda prompt: calls.append(prompt),
        [],
        code_names={"drugs": {"A": "long name " * 200}},
        prompt_token_counter=len,
        max_prompt_tokens=500,
        candidate_batch_size=1,
    )
    with pytest.raises(ValueError, match="complete interaction context"):
        model.refine_with_details({"patient_id": "test"}, [0.5])
    assert not calls


def test_batching_rejects_unbudgeted_or_invalid_size():
    with pytest.raises(ValueError, match="requires a token counter"):
        GiantMed(["A"], [[0]], lambda p: "{}", [], candidate_batch_size=1)
    with pytest.raises(ValueError, match="positive integer"):
        GiantMed(
            ["A"],
            [[0]],
            lambda p: "{}",
            [],
            candidate_batch_size=True,
            prompt_token_counter=len,
            max_prompt_tokens=1000,
        )


def test_batch_row_indices_map_to_targets_and_reject_duplicate_aliases():
    replies = iter(['{"0":0.8,"A":0.9,"1":0.7,"2":0.1}', '{"0":0.6,"1":0.55}'])
    model = GiantMed(
        ["A", "B", "C", "D"],
        np.zeros((4, 4)),
        lambda prompt: next(replies),
        [],
        prompt_token_counter=len,
        max_prompt_tokens=3000,
        candidate_batch_size=2,
    )
    result = model.refine_with_details({"patient_id": "test"}, [0.4] * 4)
    # A has conflicting aliases; the extra numeric key makes the first batch ambiguous.
    np.testing.assert_allclose(result["probabilities"], [0.4, 0.4, 0.6, 0.55])
    assert result["status"] == "partial"
    assert result["updated"] == ["C", "D"]


def test_batched_invalid_response_only_falls_back_for_that_batch():
    answers = iter(["not json", '{"0":0.8,"1":0.7}'])
    model = GiantMed(
        ["A", "B", "C", "D"],
        np.zeros((4, 4)),
        lambda prompt: next(answers),
        [],
        prompt_token_counter=len,
        max_prompt_tokens=3000,
        candidate_batch_size=2,
    )
    result = model.refine_with_details(
        {"patient_id": "test", "drugs": ["SECRET_TARGET"]}, [0.4] * 4
    )
    np.testing.assert_allclose(result["probabilities"], [0.4, 0.4, 0.8, 0.7])
    assert result["status"] == "partial"
    assert all("SECRET_TARGET" not in prompt for prompt in result["prompts"])


@pytest.mark.parametrize(
    "response", ['{"1":0.8,"2":0.9}', '{"1":0.8}', '{"0":0.8}', '{"00":0.8,"1":0.9}']
)
def test_incomplete_or_one_based_numeric_batches_fall_back(response):
    model = GiantMed(
        ["A", "B"],
        np.zeros((2, 2)),
        lambda prompt: response,
        [],
        prompt_token_counter=len,
        max_prompt_tokens=3000,
        candidate_batch_size=2,
    )
    result = model.refine_with_details({"patient_id": "test"}, [0.4, 0.4])
    np.testing.assert_allclose(result["probabilities"], [0.4, 0.4])
    assert result["status"] == "partial"
