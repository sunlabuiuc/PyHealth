import inspect
import unittest

from pyhealth.medcode.pretrained_embeddings.kg_emb.datasets import SampleKGDataset


def _toy_kg_dataset():
    """A tiny, fixed knowledge graph used across these tests: three
    entities, one relation, one triple (aspirin treats headache)."""
    entity2id = {"aspirin": 0, "headache": 1, "ibuprofen": 2}
    relation2id = {"treats": 0}
    samples = [
        {
            "triple": (0, 0, 1),
            "ground_truth_head": [0, 2],
            "ground_truth_tail": [1],
            "subsampling_weight": 1.0,
        },
    ]
    return SampleKGDataset(
        samples=samples,
        dataset_name="toy_kg",
        entity_num=len(entity2id),
        relation_num=len(relation2id),
        entity2id=entity2id,
        relation2id=relation2id,
    )


class TestKgEmbImports(unittest.TestCase):
    def test_import_pretrained_embeddings(self):
        try:
            import pyhealth.medcode.pretrained_embeddings  # noqa: F401
        except ImportError as e:
            self.fail(
                f"Importing pyhealth.medcode.pretrained_embeddings failed: {e}"
            )

    def test_model_classes_importable(self):
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import (
            ComplEx,
            DistMult,
            KGEBaseModel,
            RotatE,
            TransE,
        )

        for cls in (KGEBaseModel, TransE, RotatE, DistMult, ComplEx):
            self.assertTrue(
                isinstance(cls, type),
                msg=f"{cls} was not importable as a class",
            )

    def test_transe_uses_sample_kg_dataset(self):
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

        sig = inspect.signature(TransE.__init__)
        dataset_annotation = sig.parameters["dataset"].annotation
        self.assertEqual(
            dataset_annotation,
            SampleKGDataset,
            msg=(
                "TransE.__init__'s `dataset` parameter should be typed as "
                "SampleKGDataset, not the base SampleDataset — "
                "KGEBaseModel reads dataset.entity_num / dataset.relation_num, "
                "which only exist on SampleKGDataset"
            ),
        )


class TestKgEmbInstantiation(unittest.TestCase):
    """These tests actually build a SampleKGDataset and construct each
    model from it, so a broken __init__ (e.g. an incompatible call to
    SampleDataset.__init__) fails loudly here instead of only surfacing
    downstream during training."""

    def test_sample_kg_dataset_constructs_and_reports_correct_length(self):
        dataset = _toy_kg_dataset()
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.entity_num, 3)
        self.assertEqual(dataset.relation_num, 1)

    def test_split_works_on_a_real_sample_kg_dataset(self):
        from pyhealth.medcode.pretrained_embeddings.kg_emb.datasets import split

        entity2id = {"aspirin": 0, "headache": 1, "ibuprofen": 2}
        relation2id = {"treats": 0}
        samples = [
            {"triple": (0, 0, 1), "ground_truth_head": [0], "ground_truth_tail": [1], "subsampling_weight": 1.0},
            {"triple": (2, 0, 1), "ground_truth_head": [2], "ground_truth_tail": [1], "subsampling_weight": 1.0},
        ]
        dataset = SampleKGDataset(
            samples=samples, entity_num=3, relation_num=1,
            entity2id=entity2id, relation2id=relation2id,
        )
        train, val, test = split(dataset, ratios=[0.5, 0.5, 0.0], seed=42)
        self.assertEqual(len(train) + len(val) + len(test), 2)

    def test_each_kge_model_constructs_from_a_real_dataset(self):
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import (
            ComplEx,
            DistMult,
            KGEBaseModel,
            RotatE,
            TransE,
        )

        dataset = _toy_kg_dataset()
        for cls in (KGEBaseModel, TransE, RotatE, DistMult, ComplEx):
            model = cls(dataset, e_dim=32, r_dim=32)
            self.assertEqual(
                tuple(model.E_emb.shape),
                (3, 32),
                msg=f"{cls.__name__}'s entity embedding has the wrong shape",
            )
            self.assertEqual(
                tuple(model.R_emb.shape),
                (1, 32),
                msg=f"{cls.__name__}'s relation embedding has the wrong shape",
            )


if __name__ == "__main__":
    unittest.main()
