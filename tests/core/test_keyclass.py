import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import KeyClass
from pyhealth.weak_supervision import (
    ClinicalTextEncoder,
    KeywordLabeler,
    aggregate_votes,
    clean_note,
    confident_balanced_mask,
    nltk_english_stopwords,
    snorkel_memory_gb,
    vote_diagnostics,
)

# Small stand-in for the NLTK corpus so unit tests need no download.
STOPWORDS = ("and", "at", "by", "no", "not", "the")


def corpus_available() -> bool:
    """Returns whether NLTK and a readable English stopwords corpus are installed."""
    try:
        from nltk.corpus import stopwords

        stopwords.words("english")
    except (ImportError, LookupError, OSError):  # OSError: damaged corpus files
        return False
    return True


class MockEncoder:
    def encode(self, texts):
        return np.array(
            [[1 + t.count("heart"), 1 + t.count("lung")] for t in texts], dtype=float
        )


class TestKeyClass(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch(
            "pyhealth.weak_supervision.keyclass.nltk_english_stopwords",
            return_value=STOPWORDS,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_mining_and_boundaries(self):
        labeler = KeywordLabeler(MockEncoder(), top_k=4, min_df=1).fit(
            ["heart failure", "lung disease", "heart lung"],
            {"cardiac": "heart", "resp": "lung"},
        )
        self.assertEqual(labeler.category_ids_, ("cardiac", "resp"))
        self.assertTrue(any(lf.phrase == "heart failure" for lf in labeler.lfs_))
        votes = labeler.transform(["HEART failure", "heartland", "", "heart lung"])
        self.assertTrue((votes[0] != -1).any())
        self.assertTrue((votes[1:3] == -1).all())
        self.assertEqual(labeler.transform([]).shape, (0, len(labeler.lfs_)))
        self.assertGreater(vote_diagnostics(votes, 2)["cross_category_fraction"], 0)
        # Transform must not learn new phrases from held-out notes.
        self.assertNotIn("heartland", labeler.vectorizer_.vocabulary_)

    def test_majority_abstention_and_validation(self):
        probabilities, covered = aggregate_votes(
            np.array([[0, 1, -1], [-1, -1, -1], [1, 1, 0]]), 2
        )
        np.testing.assert_allclose(
            probabilities, [[0.5, 0.5], [0.5, 0.5], [1 / 3, 2 / 3]]
        )
        self.assertEqual(covered.tolist(), [True, False, True])
        with self.assertRaises(ValueError):
            aggregate_votes(np.array([[2]]), 2)
        p, mask = aggregate_votes(np.empty((2, 0), dtype=int), 2)
        self.assertFalse(mask.any())
        np.testing.assert_allclose(p, 0.5)

    def test_pyhealth_batch_and_gradients(self):
        samples = [
            {
                "patient_id": str(i),
                "embedding": [1.0, float(i)],
                "weak_labels": [0.2, 0.8],
            }
            for i in range(2)
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"embedding": "tensor"},
            output_schema={"weak_labels": "tensor"},
        )
        model = KeyClass(dataset, 2, ["cardiac", "resp"], hidden_dims=(8, 8, 4))
        batch = next(iter(get_dataloader(dataset, batch_size=2)))
        result = model(**batch)
        self.assertEqual(model.mode, "multilabel")
        self.assertEqual(model.label_processor.output_size(), 2)
        self.assertEqual(model.label_processor.label_vocab, {"cardiac": 0, "resp": 1})
        self.assertEqual(list(model.label_processors), ["weak_labels"])
        self.assertEqual(result["y_prob"].shape, (2, 2))
        result["loss"].backward()
        self.assertTrue(torch.isfinite(model.network[0].weight.grad).all())
        model.eval()
        self.assertNotIn("loss", model(embedding=batch["embedding"]))
        with self.assertRaises(ValueError):
            model(embedding=batch["embedding"], weak_labels=[[2.0, 0.0], [0.0, 1.0]])

    def test_clean_note_and_stopwords(self):
        text = (
            "Seen by [**First Name4 (NamePattern1) **] at [**Hospital1 18**].\nASA 81"
        )
        self.assertEqual(clean_note(text), "seen by at . asa")
        labeler = KeywordLabeler(MockEncoder(), ngram_range=(1, 2), min_df=1)
        labeler.fit(["no heart failure", "lung not clear"], {"c": "heart", "r": "lung"})
        # Stopwords are removed before n-grams are formed, as in the authors' code.
        self.assertIn("heart failure", labeler.vectorizer_.vocabulary_)
        self.assertNotIn("no", labeler.vectorizer_.vocabulary_)
        self.assertIn("lung clear", labeler.vectorizer_.vocabulary_)

    def test_missing_nltk_corpus_message(self):
        nltk_english_stopwords.cache_clear()
        self.addCleanup(nltk_english_stopwords.cache_clear)
        corpus = mock.Mock()
        corpus.words.side_effect = LookupError
        with (
            mock.patch.dict(
                "sys.modules", {"nltk.corpus": mock.Mock(stopwords=corpus)}
            ),
            self.assertRaisesRegex(LookupError, "nltk.downloader stopwords"),
        ):
            nltk_english_stopwords()

    @unittest.skipUnless(corpus_available(), "NLTK stopwords corpus not installed")
    def test_real_nltk_stopwords(self):
        nltk_english_stopwords.cache_clear()
        self.addCleanup(nltk_english_stopwords.cache_clear)
        words = nltk_english_stopwords()
        self.assertIn("not", words)
        self.assertNotIn("don't", words)  # cannot match a CountVectorizer token
        self.assertTrue(all(len(w) > 1 for w in words))

    def test_strict_min_df_and_candidate_cap(self):
        texts = ["heart failure"] * 2 + ["lung disease"] * 3
        labeler = KeywordLabeler(MockEncoder(), ngram_range=(1, 1), min_df=0.4)
        labeler.fit(texts, {"cardiac": "heart", "resp": "lung"})
        self.assertEqual(labeler.min_count_, 3)  # more than 40% of 5 notes
        self.assertEqual(set(labeler.vectorizer_.vocabulary_), {"lung", "disease"})
        with self.assertRaises(ValueError):
            KeywordLabeler(MockEncoder(), min_df=1, max_candidates=1).fit(
                texts, {"cardiac": "heart", "resp": "lung"}
            )

    def test_top_k_budget_and_vectorized_votes(self):
        texts = ["heart failure acute", "heart murmur", "lung disease", "heart lung"]
        descriptions = {"cardiac": "heart", "resp": "lung"}
        uncapped = KeywordLabeler(MockEncoder(), top_k=50, min_df=1).fit(
            texts, descriptions
        )
        pools = np.bincount([lf.category_index for lf in uncapped.lfs_], minlength=2)
        self.assertGreater(pools.max(), pools.min())
        capped = KeywordLabeler(
            MockEncoder(), top_k=50, min_df=1, equalize_top_k=True
        ).fit(texts, descriptions)
        self.assertEqual(capped.effective_top_k_, pools.min())
        votes = uncapped.transform(texts)
        counts = uncapped.vectorizer_.transform(texts)
        for j, lf in enumerate(uncapped.lfs_):
            column = uncapped.vectorizer_.vocabulary_[lf.phrase]
            expected = counts[:, column].toarray().ravel() > 0
            np.testing.assert_array_equal(votes[:, j] == lf.category_index, expected)

    def test_confident_mask_and_snorkel_guard(self):
        p = np.array([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6], [0.5, 0.5], [0.1, 0.9]])
        covered = np.array([True, True, True, False, True])
        mask = confident_balanced_mask(p, covered, 2)
        np.testing.assert_array_equal(mask, [True, False, False, False, True])
        self.assertGreater(snorkel_memory_gb(945, 5700, 19), 140)
        with self.assertRaises(MemoryError):
            aggregate_votes(
                np.array([[0, 1, -1], [1, 1, 0]]), 2, "snorkel", max_memory_gb=1e-9
            )

    def test_standardization_and_sample_weights(self):
        samples = [
            {
                "patient_id": str(i),
                "embedding": [1.0, 10.0 * i],
                "weak_labels": [0.2, 0.8],
            }
            for i in range(3)
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"embedding": "tensor"},
            output_schema={"weak_labels": "tensor"},
        )
        model = KeyClass(dataset, 2, ["cardiac", "resp"], hidden_dims=(4, 4, 4)).eval()
        x = torch.tensor([[1.0, 0.0], [1.0, 10.0], [1.0, 20.0]])
        y = torch.tensor([[0.2, 0.8], [1.0, 0.0], [0.5, 0.5]])
        torch.testing.assert_close(model(embedding=x)["logit"], model.network(x))
        model.fit_input_standardization(x)
        self.assertEqual(float(model.input_scale[0]), 1.0)  # constant dimension
        scaled = (x - x.mean(0)) / torch.tensor([1.0, x[:, 1].std(unbiased=False)])
        torch.testing.assert_close(model(embedding=x)["logit"], model.network(scaled))
        plain = model(embedding=x, weak_labels=y)["loss"]
        ones = model(embedding=x, weak_labels=y, sample_weight=torch.ones(3))["loss"]
        torch.testing.assert_close(plain, ones)
        zeros = model(embedding=x, weak_labels=y, sample_weight=torch.zeros(3))
        self.assertEqual(zeros["loss"].item(), 0.0)
        with self.assertRaises(ValueError):
            model(embedding=x, weak_labels=y, sample_weight=torch.ones(2))
        restored = KeyClass(dataset, 2, ["cardiac", "resp"], hidden_dims=(4, 4, 4))
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(
            restored.eval()(embedding=x)["logit"], model(embedding=x)["logit"]
        )

    @unittest.skipUnless(
        importlib.util.find_spec("snorkel"), "optional Snorkel not installed"
    )
    def test_real_snorkel(self):
        votes = np.tile([[0, 0, -1], [1, -1, 1], [0, 1, 0], [1, 1, 1]], (10, 1))
        votes = np.vstack([votes, [-1, -1, -1]])
        p, covered = aggregate_votes(
            votes, 2, method="snorkel", n_epochs=20, progress_bar=False
        )
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)
        self.assertFalse(covered[-1])
        np.testing.assert_allclose(p[-1], [0.5, 0.5])


def tiny_dataset():
    """A two-sample dataset with 2-dimensional embeddings and 2 categories."""
    samples = [
        {"patient_id": str(i), "embedding": [1.0, float(i)], "weak_labels": [0.5, 0.5]}
        for i in range(2)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"embedding": "tensor"},
        output_schema={"weak_labels": "tensor"},
    )


class FakeLabelModel:
    """Stands in for Snorkel's LabelModel so its code path runs without Snorkel."""

    normalize = True

    def __init__(self, cardinality, device, verbose):
        self.cardinality, self.device = cardinality, device

    def fit(self, L_train, seed, **kwargs):
        self.fit_rows, self.fit_kwargs = len(L_train), kwargs

    def predict_proba(self, L):
        scores = np.arange(1, len(L) * self.cardinality + 1, dtype=float)
        scores = scores.reshape(len(L), self.cardinality)
        return scores / scores.sum(1, keepdims=True) if self.normalize else scores


def fake_snorkel(label_model):
    """sys.modules entries that make ``snorkel.labeling.model`` importable."""
    module = types.ModuleType("snorkel.labeling.model")
    module.LabelModel = label_model
    return {
        "snorkel": types.ModuleType("snorkel"),
        "snorkel.labeling": types.ModuleType("snorkel.labeling"),
        "snorkel.labeling.model": module,
    }


class TestValidationAndOptionalPaths(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch(
            "pyhealth.weak_supervision.keyclass.nltk_english_stopwords",
            return_value=STOPWORDS,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_model_rejects_bad_configuration_and_inputs(self):
        dataset = tiny_dataset()
        with self.assertRaises(ValueError):
            KeyClass(dataset, 2, ["cardiac"])
        with self.assertRaises(ValueError):
            KeyClass(dataset, 2, ["cardiac", "cardiac"])
        with self.assertRaises(ValueError):
            KeyClass(dataset, 2, ["cardiac", "resp"], hidden_dims=(4, 0))
        model = KeyClass(dataset, 2, ["cardiac", "resp"], hidden_dims=(4,))
        with self.assertRaises(ValueError):
            model.fit_input_standardization(torch.ones(3, 5))
        with self.assertRaises(ValueError):
            model.fit_input_standardization(torch.ones(1, 2))
        with self.assertRaises(ValueError):
            model.fit_input_standardization([[1.0, float("nan")], [2.0, 3.0]])
        with self.assertRaises(ValueError):
            model(embedding=torch.ones(2))

    def test_labeler_rejects_bad_settings_and_inputs(self):
        descriptions = {"cardiac": "heart", "resp": "lung"}
        with self.assertRaises(ValueError):
            KeywordLabeler(MockEncoder(), top_k=0)
        with self.assertRaises(ValueError):
            KeywordLabeler(MockEncoder(), min_df=1.5).fit(["heart"], descriptions)
        with self.assertRaises(ValueError):
            KeywordLabeler(MockEncoder(), min_df=0).fit(["heart"], descriptions)
        labeler = KeywordLabeler(MockEncoder(), min_df=1)
        with self.assertRaises(ValueError):
            labeler.fit([], descriptions)
        with self.assertRaises(ValueError):
            labeler.fit(["heart"], {"cardiac": "heart"})
        with self.assertRaises(ValueError):
            labeler.fit(["heart"], {"cardiac": "heart", "resp": " "})
        with self.assertRaises(RuntimeError):
            KeywordLabeler(MockEncoder()).transform(["heart"])

        class ZeroEncoder:
            def encode(self, texts):
                return np.zeros((len(texts), 2))

        with self.assertRaises(ValueError):
            KeywordLabeler(ZeroEncoder(), min_df=1).fit(["heart"], descriptions)

    def test_vote_and_mask_validation(self):
        with self.assertRaises(ValueError):
            aggregate_votes(np.array([[0.5, 1.0]]), 2)
        with self.assertRaises(ValueError):
            aggregate_votes(np.array([[0, 1]]), 2, method="vote")
        with self.assertRaises(ValueError):
            confident_balanced_mask(np.ones((3, 2)), np.ones(2, dtype=bool), 4)
        with self.assertRaises(ValueError):
            confident_balanced_mask(np.ones((3, 2)), np.ones(3, dtype=bool), 1)

    def test_snorkel_path_without_snorkel_installed(self):
        votes = np.array([[0, 0, -1], [1, -1, 1], [-1, -1, -1]])
        with mock.patch.dict("sys.modules", fake_snorkel(FakeLabelModel)):
            p, covered = aggregate_votes(votes, 2, method="snorkel", n_epochs=5)
            self.assertEqual(covered.tolist(), [True, True, False])
            np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)
            np.testing.assert_allclose(p[-1], [0.5, 0.5])  # uncovered: uniform
            with self.assertRaises(ValueError):  # fewer than 3 rules
                aggregate_votes(votes[:, :2], 2, method="snorkel")

        class UnnormalizedLabelModel(FakeLabelModel):
            normalize = False

        with (
            mock.patch.dict("sys.modules", fake_snorkel(UnnormalizedLabelModel)),
            self.assertRaisesRegex(ValueError, "sum to one"),
        ):
            aggregate_votes(votes, 2, method="snorkel")
        with (
            mock.patch.dict("sys.modules", {"snorkel.labeling.model": None}),
            self.assertRaisesRegex(ImportError, "Snorkel is not installed"),
        ):
            aggregate_votes(votes, 2, method="snorkel")

    def test_nltk_stopwords_filtering_and_missing_package(self):
        nltk_english_stopwords.cache_clear()
        self.addCleanup(nltk_english_stopwords.cache_clear)
        corpus = mock.Mock()
        corpus.words.return_value = ["the", "a", "don't", "not", "the"]
        with mock.patch.dict(
            "sys.modules", {"nltk.corpus": mock.Mock(stopwords=corpus)}
        ):
            self.assertEqual(nltk_english_stopwords(), ("not", "the"))
        nltk_english_stopwords.cache_clear()
        with (
            mock.patch.dict("sys.modules", {"nltk": None, "nltk.corpus": None}),
            self.assertRaisesRegex(ImportError, "pip install"),
        ):
            nltk_english_stopwords()


class TestTrainerIntegration(unittest.TestCase):
    """Trains and evaluates KeyClass with PyHealth's Trainer and its metrics."""

    @staticmethod
    def dataset(with_gold):
        rng = np.random.default_rng(0)
        samples = []
        for i in range(12):
            category = i % 3
            weak = np.full(3, 0.1)
            weak[category] = 0.8
            sample = {
                "patient_id": str(i),
                "embedding": (np.eye(3)[category] + 0.1 * rng.normal(size=3)).tolist(),
                "weak_labels": weak.tolist(),
            }
            if with_gold:
                gold = np.eye(3)[category]
                gold[(category + 1) % 3] = i % 2  # some notes have two diagnoses
                sample["gold_labels"] = gold.tolist()
            samples.append(sample)
        output_schema = {"weak_labels": "tensor"}
        if with_gold:
            output_schema["gold_labels"] = "tensor"
        return create_sample_dataset(
            samples=samples,
            input_schema={"embedding": "tensor"},
            output_schema=output_schema,
        )

    def train_and_evaluate(self, with_gold):
        from pyhealth.trainer import Trainer

        torch.manual_seed(0)
        dataset = self.dataset(with_gold)
        model = KeyClass(
            dataset,
            3,
            ["cardiac", "resp", "renal"],
            hidden_dims=(8,),
            gold_key="gold_labels" if with_gold else None,
        )
        loader = get_dataloader(dataset, batch_size=4)
        trainer = Trainer(
            model=model,
            metrics=["f1_samples", "roc_auc_macro"],
            enable_logging=False,
        )
        trainer.train(train_dataloader=loader, val_dataloader=loader, epochs=2)
        y_true, _, _ = trainer.inference(loader)
        return trainer.evaluate(loader), y_true

    def test_trainer_with_weak_labels_only(self):
        scores, y_true = self.train_and_evaluate(with_gold=False)
        # Weak labels at or above 1/3 become the binary y_true for the metrics.
        np.testing.assert_array_equal(y_true, np.eye(3)[np.arange(12) % 3])
        for key in ("loss", "f1_samples", "roc_auc_macro"):
            self.assertTrue(np.isfinite(scores[key]), key)

    def test_trainer_scores_against_gold_labels(self):
        scores, y_true = self.train_and_evaluate(with_gold=True)
        expected = np.eye(3)[np.arange(12) % 3]
        expected[np.arange(12), (np.arange(12) + 1) % 3] = np.arange(12) % 2
        np.testing.assert_array_equal(y_true, expected)
        for key in ("loss", "f1_samples", "roc_auc_macro"):
            self.assertTrue(np.isfinite(scores[key]), key)

    def test_gold_labels_do_not_change_the_loss(self):
        dataset = self.dataset(with_gold=True)
        batch = next(iter(get_dataloader(dataset, batch_size=4)))
        model = KeyClass(dataset, 3, ["a", "b", "c"], gold_key="gold_labels").eval()
        weak_only = {k: v for k, v in batch.items() if k != "gold_labels"}
        torch.testing.assert_close(model(**batch)["loss"], model(**weak_only)["loss"])
        with self.assertRaises(ValueError):
            model(**{**batch, "gold_labels": batch["gold_labels"] * 0.5})
        with self.assertRaises(ValueError):
            model(**{**batch, "gold_labels": batch["gold_labels"][:, :2]})


class TestClinicalTextEncoder(unittest.TestCase):
    """Runs the encoder on a tiny randomly initialized BERT; nothing is downloaded."""

    VOCAB = (
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "heart", "failure", "renal",
        "sepsis", "pneumonia", "acute", "chronic", "the", "patient", "with",
    )  # fmt: skip

    @classmethod
    def setUpClass(cls):
        from transformers import BertConfig, BertModel, BertTokenizer

        cls._tmp = tempfile.TemporaryDirectory()
        path = Path(cls._tmp.name)
        (path / "vocab.txt").write_text("\n".join(cls.VOCAB) + "\n")
        BertTokenizer(str(path / "vocab.txt")).save_pretrained(path)
        torch.manual_seed(0)
        config = BertConfig(
            vocab_size=len(cls.VOCAB),
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=32,
        )
        BertModel(config).save_pretrained(path)
        cls.model_dir = str(path)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_output_shape_and_empty_input(self):
        encoder = ClinicalTextEncoder(self.model_dir, batch_size=2)
        out = encoder.encode(["heart failure", "sepsis", "acute renal failure"])
        self.assertEqual(out.shape, (3, 8))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(encoder.encode([]).shape, (0, 8))

    def test_padding_does_not_change_embeddings(self):
        texts = ["sepsis", "the patient with acute chronic renal failure", "heart"]
        one_at_a_time = ClinicalTextEncoder(self.model_dir, batch_size=1)
        padded_batch = ClinicalTextEncoder(self.model_dir, batch_size=3)
        np.testing.assert_allclose(
            one_at_a_time.encode(texts), padded_batch.encode(texts), atol=1e-5
        )

    def test_truncation_ignores_text_past_max_length(self):
        texts = ["heart failure sepsis", "heart failure pneumonia acute"]
        short = ClinicalTextEncoder(self.model_dir, max_length=4).encode(texts)
        np.testing.assert_allclose(short[0], short[1], atol=1e-6)
        full = ClinicalTextEncoder(self.model_dir, max_length=32).encode(texts)
        self.assertFalse(np.allclose(full[0], full[1]))

    def test_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError):
            ClinicalTextEncoder(self.model_dir, batch_size=0)
        with self.assertRaises(ValueError):
            ClinicalTextEncoder(self.model_dir, max_length=1)
