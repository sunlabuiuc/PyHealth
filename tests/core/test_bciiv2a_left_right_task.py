"""Tests for the BCI Competition IV 2a left- vs right-hand imagery task."""

import shutil
import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import (
    BCICompetitionIV2aDataset,
    get_dataloader,
    split_by_strat_fold,
)
from pyhealth.models import SparcNet
from pyhealth.tasks import BCIIV2aLeftRightImagery
from tests.core.test_bciiv2a_task import FS, write_bnci_session


class TestBCIIV2aLeftRightImagery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        root = cls.tmp / "bnci"
        root.mkdir()
        for session in ("T", "E"):
            # 8 trials per session, labels cycle left, right, feet, tongue;
            # trial 1 (right hand) is artifact-marked.
            write_bnci_session(root / f"A01{session}.mat", n_trials=8)
        cls.dataset = BCICompetitionIV2aDataset(
            root=str(root),
            subjects=[1],
            metadata_cache_dir=str(cls.tmp / "meta"),
            cache_dir=str(cls.tmp / "cache"),
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_keeps_only_left_and_right(self):
        patient = self.dataset.get_patient("A01")
        samples = BCIIV2aLeftRightImagery()(patient)
        # Per session: trials 0, 4 (left) and 5 (right); trial 1 is an artifact.
        self.assertEqual(sorted(s["label"] for s in samples), [0, 0, 0, 0, 1, 1])
        self.assertEqual({s["trial"] for s in samples}, {0, 4, 5})
        kept = BCIIV2aLeftRightImagery(drop_artifacts=False)(patient)
        self.assertEqual(len(kept), 8)
        self.assertEqual(kept[0]["signal"].shape, (22, 4 * FS))

    def test_set_task_split_and_model(self):
        samples = self.dataset.set_task(BCIIV2aLeftRightImagery(), num_workers=1)
        try:
            self.assertEqual(len(samples), 6)
            train, _, test = split_by_strat_fold(
                samples,
                train_folds=(0,),
                val_folds=(),
                test_folds=(1,),
                fold_field="session_fold",
            )
            self.assertEqual((len(train), len(test)), (3, 3))
            model = SparcNet(dataset=samples)
            self.assertEqual(model.mode, "binary")
            batch = next(iter(get_dataloader(train, batch_size=3)))
            out = model(**batch)
            self.assertEqual(tuple(out["y_prob"].shape), (3, 1))
        finally:
            samples.close()


if __name__ == "__main__":
    unittest.main()
