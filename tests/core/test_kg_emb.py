import inspect
import unittest

from pyhealth.datasets import SampleDataset


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

    def test_transe_uses_sample_dataset(self):
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

        sig = inspect.signature(TransE.__init__)
        dataset_annotation = sig.parameters["dataset"].annotation
        self.assertEqual(
            dataset_annotation,
            SampleDataset,
            msg=(
                "TransE.__init__'s `dataset` parameter is not annotated as "
                "SampleDataset — regression check for the "
                "SampleBaseDataset -> SampleDataset rename"
            ),
        )


if __name__ == "__main__":
    unittest.main()