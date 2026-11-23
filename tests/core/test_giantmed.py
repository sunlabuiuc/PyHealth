"""Tests for GiantMed's retrieval, prompt and probability refinement."""

import numpy as np
import pytest

from pyhealth.models.giantmed import GiantMed, MediEvidenceFilter


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
    assert result["status"] == "no_updates"


@pytest.mark.parametrize("mode", ["plain", "budget", "batch"])
def test_prompt_modes_share_clinical_guidance(mode):
    options = {}
    if mode != "plain":
        options.update(prompt_token_counter=len, max_prompt_tokens=10000)
    if mode == "batch":
        options["candidate_batch_size"] = 1
    model = GiantMed(["A"], [[0]], lambda p: '{"A":0.8}', [], **options)
    prompt = model.refine_with_details({"patient_id": "p"}, [0.5])["prompt"]
    for instruction in (
        "clinical pharmacy expert",
        "[0.3, 0.7)",
        "Raise",
        "lower unsupported",
        "serious interactions",
    ):
        assert instruction in prompt


@pytest.mark.parametrize(
    "response",
    [
        'Result: {"A":"0.8"}',
        '<think>ignore {"A":0.1}</think>\n```json\n{"A":0.8}\n```',
    ],
)
def test_response_with_surrounding_text_and_numeric_string(response):
    model = GiantMed(["A"], [[0]], lambda p: response, [])
    result = model.refine_with_details({"patient_id": "p"}, [0.5])
    assert result["probabilities"].tolist() == [0.8]
    assert result["status"] == "complete"


def test_empty_response_status_and_code_validation():
    model = GiantMed(["A"], [[0]], lambda p: "{}", [])
    assert (
        model.refine_with_details({"patient_id": "p"}, [0.5])["status"] == "no_updates"
    )
    with pytest.raises(ValueError, match="strings"):
        GiantMed([1], [[0]], lambda p: "{}", [])
    with pytest.raises(ValueError, match="numeric medication codes"):
        GiantMed(
            ["1"],
            [[0]],
            lambda p: "{}",
            [],
            prompt_token_counter=len,
            max_prompt_tokens=1000,
            candidate_batch_size=1,
        )


def test_medi_status_distinguishes_missing_mapping_and_rejection():
    filt = MediEvidenceFilter(
        {("p", "v", "A"): {"ingredient"}, ("p", "v", "B"): set()},
        {"ingredient": {"123"}},
        history_rescue=False,
    )
    query = {"conditions": [["456"]]}
    neighbour = {"patient_id": "p", "visit_id": "v"}
    assert filt.evidence_status(query, neighbour, "A") == "indication_mismatch"
    assert filt.evidence_status(query, neighbour, "B") == "missing_ingredients"
    assert filt.evidence_status(query, neighbour, "C") == "missing_mapping"
    query["conditions"] = [["123"]]
    assert filt.evidence_status(query, neighbour, "A") == "indication_match"


def test_budget_protects_similar_visits_and_interactions_from_long_history():
    codes = [f"D{i:02}" for i in range(20)]
    matrix = np.ones((20, 20)) - np.eye(20)
    training = [
        {
            "patient_id": f"p{i}",
            "visit_id": "v",
            "conditions": [["X"]],
            "procedures": [[]],
            "drugs": codes,
        }
        for i in range(3)
    ]
    query = {
        "patient_id": "query",
        "conditions": [["X"]],
        "procedures": [[]],
        "drugs_hist": [["A" * 40] * 12 for _ in range(20)] + [[]],
    }
    counts = []
    for budget in (3000, 4000, 6000):
        model = GiantMed(
            codes,
            matrix,
            lambda p: "{}",
            training,
            prompt_token_counter=len,
            max_prompt_tokens=budget,
        )
        report = model.refine_with_details(query, [0.5] * 20)["prompt_budget"]
        counts.append((report["similar_visits"][0], report["ddi_pairs"][0]))
    assert all(similar > 0 and ddi > 0 for similar, ddi in counts)
    assert counts == sorted(counts)
    assert all(a[1] <= b[1] for a, b in zip(counts, counts[1:]))


def test_batch_preserves_similar_visit_before_long_history():
    train = [
        {
            "patient_id": "train",
            "visit_id": "1",
            "conditions": [["X"]],
            "procedures": [[]],
            "drugs": ["A"],
        }
    ]
    query = {
        "patient_id": "query",
        "conditions": [["X"]],
        "procedures": [[]],
        "drugs_hist": [["long history" * 60]] * 10 + [[]],
    }
    model = GiantMed(
        ["A"],
        [[0]],
        lambda prompt: '{"0": 0.8}',
        train,
        candidate_batch_size=1,
        prompt_token_counter=len,
        max_prompt_tokens=2200,
    )
    budget = model.refine_with_details(query, [0.5])["prompt_budget"]
    assert budget["support_per_batch"][0]["similar_visits"] == [1, 1]
    assert budget["support_per_batch"][0]["history_visits"][0] < 10


def test_medi_uses_same_ndc_label_mapping_as_task(tmp_path):
    (tmp_path / "prescriptions.csv").write_text(
        "subject_id,hadm_id,ndc\n1,11,123\n1,11,456\n2,22,123\n"
    )
    (tmp_path / "medi.csv").write_text(
        "RXCUI,CODE,SAB,MEDI1,MEDI1_HPS,MEDI2_HPS\n"
        "ingredient,25000,ICD9CM,TRUE,FALSE,FALSE\n"
    )
    (tmp_path / "ndc.txt").write_text("{'123': 'clinical', '456': 'missing'}")
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
        ndc_to_atc3=lambda ndc: {"123": ["B01A", "C01A"], "456": ["D01A"]}[ndc],
    )
    query = {"conditions": [["25000"]]}
    visit = {"patient_id": "1", "visit_id": "11"}
    assert filt.evidence_status(query, visit, "B01A") == "indication_match"
    assert filt.evidence_status(query, visit, "C01A") == "indication_match"
    assert filt.evidence_status(query, visit, "D01A") == "missing_ingredients"
    assert filt.evidence_status(query, visit, "A10A") == "missing_mapping"
    assert (
        filt.evidence_status(query, {"patient_id": "2", "visit_id": "22"}, "B01A")
        == "missing_mapping"
    )


def test_reference_ndc_mapping_matches_task_and_excludes_unmapped(tmp_path):
    from examples.drug_recommendation.giantmed_mimic3 import load_reference_ndc_atc3
    from pyhealth.tasks import DrugRecommendationMIMIC3

    ndc_path = tmp_path / "ndc.txt"
    atc_path = tmp_path / "atc.csv"
    ndc_path.write_text("{'111': '1', '222': '2', 'bad': '999'}")
    atc_path.write_text("RXCUI,ATC4\n1,A10AB\n1,B01AA\n2,C03CA\n")
    mapping = load_reference_ndc_atc3(ndc_path, atc_path)
    assert mapping == {"111": "A10A", "222": "C03C"}
    task = DrugRecommendationMIMIC3(ndc_to_atc3=mapping)
    assert task.map_drugs(["111", "bad", "111", "222"]) == ["A10A", "C03C"]
    assert task.map_drugs(["bad"]) == []
    assert task.ndc_to_atc3 == mapping


def test_mimic3_task_applies_custom_mapping_to_labels_and_history():
    from types import SimpleNamespace

    import polars as pl

    from pyhealth.tasks import DrugRecommendationMIMIC3

    class Patient:
        patient_id = "p1"

        def get_events(self, event_type, filters=None, return_df=False):
            if event_type == "admissions":
                return [SimpleNamespace(hadm_id="v1"), SimpleNamespace(hadm_id="v2")]
            visit = filters[0][2]
            values = {
                "diagnoses_icd": ("diagnoses_icd/icd9_code", ["123"]),
                "procedures_icd": ("procedures_icd/icd9_code", ["42"]),
                "prescriptions": (
                    "prescriptions/ndc",
                    ["111", "bad"] if visit == "v1" else ["222"],
                ),
            }
            name, rows = values[event_type]
            return pl.DataFrame({name: rows})

    task = DrugRecommendationMIMIC3(ndc_to_atc3={"111": "A10A", "222": "C03C"})
    first, second = task(Patient())
    assert first["drugs"] == ["A10A"]
    assert second["drugs"] == ["C03C"]
    assert second["drugs_hist"] == [["A10A"], []]


def test_ndc_coverage_excludes_missing_codes(tmp_path):
    import gzip

    from examples.drug_recommendation.giantmed_mimic3 import ndc_coverage

    path = tmp_path / "prescriptions.csv.gz"
    with gzip.open(path, "wt") as stream:
        stream.write("ndc\n111\n222\n0\n111\n")
    assert ndc_coverage(path, {"111": "A10A"}) == (2, 3)


def test_decision_change_summary_counts_against_recorded_labels():
    from examples.drug_recommendation.giantmed_mimic3 import summarize_decision_changes

    target = [[1, 0, 1, 0, 1]]
    baseline = [[0.4, 0.4, 0.6, 0.6, 0.4]]
    refined = [[0.5, 0.8, 0.2, 0.1, 0.3]]
    assert summarize_decision_changes(target, baseline, refined) == {
        "added_recorded": 1,
        "added_unrecorded": 1,
        "removed_recorded": 1,
        "removed_unrecorded": 1,
    }
    with pytest.raises(ValueError, match="matching shapes"):
        summarize_decision_changes(target, baseline, [[0.1]])


@pytest.mark.parametrize("mode", ["plain", "budget", "batch"])
def test_custom_instruction_template_preserves_prompt_contract(mode):
    options = {"prompt_template": "CUSTOM guidance for {boundary}: be conservative."}
    if mode != "plain":
        options.update(prompt_token_counter=len, max_prompt_tokens=10000)
    if mode == "batch":
        options["candidate_batch_size"] = 1
    response = '{"0":0.8}' if mode == "batch" else '{"A":0.8}'
    model = GiantMed(["A"], [[0]], lambda p: response, [], **options)
    result = model.refine_with_details(
        {"patient_id": "p", "drugs": ["SECRET_TARGET"]}, [0.5]
    )
    assert result["probabilities"].tolist() == [0.8]
    assert "CUSTOM guidance for [0.3, 0.7)" in result["prompt"]
    assert "JSON object" in result["prompt"]
    assert "SECRET_TARGET" not in result["prompt"]
    if mode != "plain":
        assert len(result["prompt"]) <= 10000
    if mode == "batch":
        assert "Only the first 1 rows are targets" in result["prompt"]


@pytest.mark.parametrize(
    "template",
    [
        "",
        "no boundary",
        "bad {secret}",
        "{boundary!r}",
        "{boundary:>10}",
        "{boundary}\nextra",
        "{boundary}\n",
        "{boundary}\r",
        "{boundary}\r\n",
        "{boundary}\u2028",
        "{boundary}\vextra",
        "{boundary.__class__}",
        "{boundary",
    ],
)
def test_invalid_prompt_template_rejected(template):
    with pytest.raises(ValueError, match="prompt_template"):
        GiantMed(["A"], [[0]], lambda p: "{}", [], prompt_template=template)


@pytest.mark.parametrize("suffix", ["\n", "\r", "\r\n", "\u2028"])
def test_trailing_line_break_rejected_with_budget(suffix):
    with pytest.raises(ValueError, match="single-line"):
        GiantMed(
            ["A"],
            [[0]],
            lambda p: "{}",
            [],
            prompt_template="Assess {boundary}." + suffix,
            prompt_token_counter=len,
            max_prompt_tokens=4096,
        )


@pytest.mark.parametrize(
    "size, expected", [("4", "requires --prompt-tokenizer"), ("0", "must be positive")]
)
def test_example_rejects_invalid_batch_configuration(
    monkeypatch, capsys, size, expected
):
    import sys

    from examples.drug_recommendation.giantmed_mimic3 import main

    monkeypatch.setattr(sys, "argv", ["example", "--candidate-batch-size", size])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert expected in capsys.readouterr().err


def test_prompt_metadata_has_stable_patient_independent_hash():
    import hashlib

    template = "Risk boundary {boundary}; evaluate interactions."
    model = GiantMed(["A"], [[0]], lambda p: "{}", [], prompt_template=template)
    metadata = model.prompt_metadata()
    assert metadata["template"] == template
    assert metadata["template_sha256"] == hashlib.sha256(template.encode()).hexdigest()
    assert model._instructions() == "Risk boundary [0.3, 0.7); evaluate interactions."


def test_medi_matches_prefixed_icd_versions_without_cross_system_collision():
    filt = MediEvidenceFilter(
        {("p", "v", "A"): {"ingredient"}},
        {"ingredient": {"25000", "E119"}},
        ingredient_icd10_diagnoses={"ingredient": {"E119", "I10"}},
        history_rescue=False,
    )
    neighbour = {"patient_id": "p", "visit_id": "v"}
    assert (
        filt.evidence_status({"conditions": [["9_250.00"]]}, neighbour, "A")
        == "indication_match"
    )
    assert (
        filt.evidence_status({"conditions": [["10_E11.9"]]}, neighbour, "A")
        == "indication_match"
    )
    assert (
        filt.evidence_status({"conditions": [["9_I10"]]}, neighbour, "A")
        == "indication_mismatch"
    )
    assert (
        filt.evidence_status({"conditions": [["10_25000"]]}, neighbour, "A")
        == "indication_mismatch"
    )
    assert (
        filt.evidence_status({"conditions": [["11_E119"]]}, neighbour, "A")
        == "no_supported_diagnosis"
    )
    assert (
        filt.evidence_status({"conditions": [["25000"]]}, neighbour, "A")
        == "indication_match"
    )
    assert (
        filt.evidence_status({"conditions": [["9_25000", "10_I10"]]}, neighbour, "A")
        == "indication_match"
    )


def test_medi_icd10_missing_indications_is_not_mismatch():
    filt = MediEvidenceFilter(
        {("p", "v", "A"): {"ingredient"}},
        {"ingredient": {"25000"}},
        history_rescue=False,
    )
    neighbour = {"patient_id": "p", "visit_id": "v"}
    assert (
        filt.evidence_status({"conditions": [["10_E119"]]}, neighbour, "A")
        == "missing_indications"
    )


def test_medi_ancestor_matching_does_not_cross_code_systems():
    filt = MediEvidenceFilter(
        {("p", "v", "A"): {"ingredient"}},
        {"ingredient": {"250"}},
        ancestor_match=True,
        ingredient_icd10_diagnoses={"ingredient": {"E11"}},
        history_rescue=False,
    )
    neighbour = {"patient_id": "p", "visit_id": "v"}
    assert filt({"conditions": [["9_25000"]]}, neighbour, "A")
    assert filt({"conditions": [["10_E119"]]}, neighbour, "A")
    assert not filt({"conditions": [["9_E119"]]}, neighbour, "A")


def test_medi_loads_icd9_and_icd10_separately_from_files(tmp_path):
    (tmp_path / "prescriptions.csv").write_text("subject_id,hadm_id,ndc\n1,11,123\n")
    (tmp_path / "medi.csv").write_text(
        "RXCUI,CODE,SAB,MEDI1,MEDI1_HPS,MEDI2_HPS\n"
        "ing,250.00,ICD9CM,TRUE,FALSE,FALSE\n"
        "ing,E11.9,ICD10CM,TRUE,FALSE,FALSE\n"
        "ing,I10,ICD10CM,FALSE,FALSE,FALSE\n"
    )
    (tmp_path / "ndc.txt").write_text("{'123':'rx'}")
    (tmp_path / "atc.csv").write_text("RXCUI,ATC4\nrx,A10AB\n")
    (tmp_path / "cache.json").write_text('{"rx":{"ingredients":[{"rxcui":"ing"}]}}')
    filt = MediEvidenceFilter.from_reference_files(
        [{"patient_id": "1", "visit_id": "11"}],
        tmp_path / "prescriptions.csv",
        tmp_path / "medi.csv",
        tmp_path / "ndc.txt",
        tmp_path / "atc.csv",
        tmp_path / "cache.json",
        profile="medi1",
        history_rescue=False,
    )
    neighbour = {"patient_id": "1", "visit_id": "11"}
    assert filt({"conditions": [["9_25000"]]}, neighbour, "A10A")
    assert filt({"conditions": [["10_E119"]]}, neighbour, "A10A")
    assert not filt({"conditions": [["10_I10"]]}, neighbour, "A10A")


def test_mimic4_task_custom_mapping_preserves_icd_versions_and_history():
    from types import SimpleNamespace

    import polars as pl

    from pyhealth.tasks import DrugRecommendationMIMIC4

    class Patient:
        patient_id = "p1"

        def get_events(self, event_type, filters=None, return_df=False):
            if event_type == "admissions":
                return [SimpleNamespace(hadm_id="v1"), SimpleNamespace(hadm_id="v2")]
            visit = filters[0][2]
            if event_type in ("diagnoses_icd", "procedures_icd"):
                version, code = ("9", "25000") if visit == "v1" else ("10", "E119")
                return pl.DataFrame(
                    {
                        f"{event_type}/icd_version": [version],
                        f"{event_type}/icd_code": [code],
                    }
                )
            return pl.DataFrame(
                {"prescriptions/ndc": ["111", "bad"] if visit == "v1" else ["222"]}
            )

    task = DrugRecommendationMIMIC4(ndc_to_atc3={"111": "A10A", "222": "C03C"})
    first, second = task(Patient())
    assert first["drugs"] == ["A10A"] and second["drugs"] == ["C03C"]
    assert second["conditions"] == [["9_25000"], ["10_E119"]]
    assert second["drugs_hist"] == [["A10A"], []]
