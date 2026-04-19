import torch

from pyhealth.datasets import create_sample_dataset
from pyhealth.models import WordBasisLinearModel


INPUT_DIM = 8
FEATURE_KEY = "embedding"
LABEL_KEY = "label"


def make_test_dataset():
    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.0, 0.1, 0.2, 0.3],
            "label": 1,
        },
        {
            "patient_id": "patient-1",
            "visit_id": "visit-1",
            "embedding": [0.3, 0.1, 0.0, 0.5, 0.2, 0.1, 0.4, 0.2],
            "label": 0,
        },
        {
            "patient_id": "patient-2",
            "visit_id": "visit-2",
            "embedding": [0.0, 0.4, 0.2, 0.1, 0.6, 0.3, 0.2, 0.1],
            "label": 1,
        },
    ]

    dataset = create_sample_dataset(
        samples=samples,
        input_schema={FEATURE_KEY: "tensor"},
        output_schema={LABEL_KEY: "binary"},
        dataset_name="word_basis_linear_model_test",
    )
    return dataset


def make_model():
    dataset = make_test_dataset()
    model = WordBasisLinearModel(
        dataset=dataset,
        input_dim=INPUT_DIM,
        feature_key=FEATURE_KEY,
        ridge_lambda=1e-4,
    )
    return model


def make_batch():
    x = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.0, 0.1, 0.2, 0.3],
            [0.3, 0.1, 0.0, 0.5, 0.2, 0.1, 0.4, 0.2],
            [0.0, 0.4, 0.2, 0.1, 0.6, 0.3, 0.2, 0.1],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    return x, y


def make_word_embeddings():
    # 6 words, each embedded in the same 8-dim space as the classifier weights
    return torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.1, 0.3, 0.4, 0.5, 0.0, 0.1, 0.2],
            [0.1, 0.3, 0.2, 0.0, 0.4, 0.5, 0.2, 0.1],
        ],
        dtype=torch.float32,
    )


def test_model_instantiation():
    model = make_model()
    assert isinstance(model, WordBasisLinearModel)
    assert model.input_dim == INPUT_DIM
    assert model.feature_key == FEATURE_KEY
    assert model.label_key == LABEL_KEY
    assert model.classifier.bias is None


def test_forward_returns_expected_keys_and_shapes():
    model = make_model()
    x, y = make_batch()

    output = model(**{FEATURE_KEY: x, LABEL_KEY: y})

    assert "loss" in output
    assert "y_prob" in output
    assert "y_true" in output
    assert "logit" in output

    assert output["logit"].shape == (3, 1)
    assert output["y_prob"].shape == (3, 1)
    assert output["y_true"].shape == (3, 1)
    assert output["loss"].ndim == 0


def test_backward_computes_gradients():
    model = make_model()
    x, y = make_batch()

    output = model(**{FEATURE_KEY: x, LABEL_KEY: y})
    output["loss"].backward()

    assert model.classifier.weight.grad is not None
    assert model.classifier.weight.grad.shape == (1, INPUT_DIM)


def test_forward_from_embedding_runs():
    model = make_model()
    x, y = make_batch()

    output = model.forward_from_embedding(feature_embeddings=x, y=y)

    assert "loss" in output
    assert "logit" in output
    assert output["logit"].shape == (3, 1)
    assert output["y_prob"].shape == (3, 1)


def test_get_classifier_weight_shape():
    model = make_model()
    beta = model.get_classifier_weight()
    assert beta.shape == (INPUT_DIM,)


def test_fit_word_basis_and_reconstruct_shapes():
    model = make_model()
    x, y = make_batch()
    _ = model(**{FEATURE_KEY: x, LABEL_KEY: y})

    word_embeddings = make_word_embeddings()
    coeffs = model.fit_word_basis(word_embeddings)

    assert coeffs.ndim == 1
    assert coeffs.shape[0] == word_embeddings.shape[0]

    beta_hat = model.reconstruct_from_word_basis(word_embeddings, coeffs)
    assert beta_hat.shape == (INPUT_DIM,)


def test_compute_word_basis_cosine_similarity_runs():
    model = make_model()
    x, y = make_batch()
    _ = model(**{FEATURE_KEY: x, LABEL_KEY: y})

    word_embeddings = make_word_embeddings()
    coeffs = model.fit_word_basis(word_embeddings)
    cosine = model.compute_word_basis_cosine_similarity(word_embeddings, coeffs)

    assert cosine.ndim == 0
    assert torch.isfinite(cosine)


def test_explain_words_returns_pairs():
    model = make_model()
    x, y = make_batch()
    _ = model(**{FEATURE_KEY: x, LABEL_KEY: y})

    word_embeddings = make_word_embeddings()
    word_list = ["word_a", "word_b", "word_c", "word_d", "word_e", "word_f"]

    pairs = model.explain_words(word_embeddings, word_list)

    assert isinstance(pairs, list)
    assert len(pairs) == len(word_list)
    assert isinstance(pairs[0][0], str)
    assert isinstance(pairs[0][1], float)