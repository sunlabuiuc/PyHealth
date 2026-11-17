"""Tests for the SleepKD distillation model. All data is synthetic."""

import copy
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import DeepSleepNet, SleepKD
from pyhealth.trainer import Trainer

SEQ_LEN = 4


def _dataset(n_samples=4):
    rng = np.random.default_rng(0)
    samples = [
        {
            "patient_id": str(i),
            "signal": rng.standard_normal((SEQ_LEN, 1, 3000)).astype(np.float32),
            "label": rng.integers(0, 5, size=SEQ_LEN),
        }
        for i in range(n_samples)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema={"label": ("tensor", {"dtype": torch.long})},
    )


class TestSleepKD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.dataset = _dataset()
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=2)))

    def _teacher(self):
        return DeepSleepNet(dataset=self.dataset, n_conv=2, hidden_dim=16)

    def _student(self):
        return DeepSleepNet(dataset=self.dataset, n_conv=1, hidden_dim=8)

    def _model(self, teacher=None, student=None, **kwargs):
        return SleepKD(
            dataset=self.dataset,
            teacher=teacher or self._teacher(),
            student=student or self._student(),
            **kwargs,
        )

    def test_loss_is_the_weighted_sum_of_the_terms(self):
        model = self._model(alpha=0.3, beta=0.2, gamma=0.4, delta=0.1)
        out = model(**self.batch)
        expected = (
            0.3 * out["loss_epoch"]
            + 0.2 * out["loss_seq"]
            + 0.4 * out["loss_soft"]
            + 0.1 * out["loss_hard"]
        )
        torch.testing.assert_close(out["loss"].detach(), expected)

    def test_outputs_are_the_students_per_epoch_predictions(self):
        model = self._model().eval()
        out = model(**self.batch)
        student = model.student.encode(self.batch["signal"])["logit"].flatten(0, 1)
        torch.testing.assert_close(out["logit"], student)
        self.assertEqual(out["y_prob"].shape, (2 * SEQ_LEN, 5))
        self.assertEqual(out["y_true"].shape, (2 * SEQ_LEN,))

    def test_hard_and_soft_terms_follow_eq_3_and_4(self):
        model = self._model().eval()
        out = model(**self.batch)
        x, y = self.batch["signal"], self.batch["label"].flatten()
        s = model.student.encode(x)["logit"].flatten(0, 1)
        t = model.teacher.encode(x)["logit"].flatten(0, 1)
        p_t = t.softmax(-1)
        kl = (p_t * (p_t.log() - s.log_softmax(-1))).sum(-1).mean()
        torch.testing.assert_close(out["loss_soft"], kl)
        torch.testing.assert_close(out["loss_hard"], F.cross_entropy(s, y))

    def test_kl_feature_loss_is_the_mean_kl_per_epoch(self):
        model = self._model(feature_loss="kl").eval()
        out = model(**self.batch)
        t = model.teacher.encode(self.batch["signal"])
        s = model.student.encode(self.batch["signal"])
        for term, key in (
            ("loss_epoch", "epoch_feature"),
            ("loss_seq", "sequence_feature"),
        ):
            p_t = model._align(key, t[key]).softmax(-1)
            kl = (p_t * (p_t.log() - s[key].log_softmax(-1))).sum(-1)
            self.assertEqual(kl.shape, (2, SEQ_LEN))
            torch.testing.assert_close(out[term], kl.mean())

    def test_copy_of_the_teacher_has_no_distillation_loss(self):
        teacher = self._teacher()
        for feature_loss in ("mse", "kl"):
            with self.subTest(feature_loss=feature_loss):
                model = self._model(
                    teacher=teacher,
                    student=copy.deepcopy(teacher),
                    feature_loss=feature_loss,
                ).eval()
                out = model(**self.batch)
                for term in ("loss_epoch", "loss_seq", "loss_soft"):
                    self.assertAlmostEqual(out[term].item(), 0.0, places=5)

    def test_maxpool_aligns_teacher_features_to_the_student(self):
        model = self._model()
        t = model.teacher.encode(self.batch["signal"])
        s = model.student.encode(self.batch["signal"])
        for key in ("epoch_feature", "sequence_feature"):
            aligned = model._align(key, t[key])
            self.assertEqual(aligned.shape, s[key].shape)
        # Sequence features: 32 teacher entries max-pooled in pairs to 16.
        seq = t["sequence_feature"]
        torch.testing.assert_close(
            model._align("sequence_feature", seq),
            seq.view(2, SEQ_LEN, 16, 2).amax(-1),
        )
        self.assertEqual(len(list(model.aligners.parameters())), 0)

    def test_slidingmax_takes_the_max_of_overlapping_windows(self):
        model = self._model(align="slidingmax")
        t = model.teacher.encode(self.batch["signal"])
        s = model.student.encode(self.batch["signal"])
        for key in ("epoch_feature", "sequence_feature"):
            self.assertEqual(model._align(key, t[key]).shape, s[key].shape)
        # Sequence features: entry i is the max of teacher entries i..i+16.
        seq = t["sequence_feature"]
        expected = seq.unfold(-1, 17, 1).amax(-1)
        torch.testing.assert_close(model._align("sequence_feature", seq), expected)
        self.assertEqual(len(list(model.aligners.parameters())), 0)

    def test_only_the_student_and_conv_aligners_learn(self):
        model = self._model(align="conv")
        model(**self.batch)["loss"].backward()
        self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
        self.assertTrue(all(p.grad is not None for p in model.aligners.parameters()))
        self.assertIsNotNone(model.student.fc.weight.grad)

    def test_teacher_stays_in_eval_mode(self):
        model = self._model()
        model.train()
        self.assertTrue(model.student.training)
        self.assertFalse(model.teacher.training)

    def test_trainer_scores_the_student(self):
        model = self._model()
        trainer = Trainer(model=model, metrics=["accuracy"], enable_logging=False)
        loader = get_dataloader(self.dataset, batch_size=2)
        teacher_before = copy.deepcopy(model.teacher.state_dict())
        trainer.train(train_dataloader=loader, epochs=1)
        for key, value in model.teacher.state_dict().items():
            torch.testing.assert_close(value, teacher_before[key], msg=key)
        self.assertEqual(set(trainer.evaluate(loader)), {"accuracy", "loss"})

    def test_rejects_bad_settings(self):
        with self.assertRaises(ValueError):
            self._model(alpha=0, beta=0, gamma=0, delta=0)
        with self.assertRaises(ValueError):
            self._model(alpha=-1.0)
        with self.assertRaises(ValueError):
            self._model(feature_loss="l1")
        with self.assertRaises(ValueError):
            self._model(align="avgpool")
        for align in ("maxpool", "slidingmax"):
            with self.assertRaises(ValueError):
                # Teacher features smaller than the student's.
                self._model(
                    teacher=self._student(), student=self._teacher(), align=align
                )
        with self.assertRaises(TypeError):
            self._model(teacher=torch.nn.Linear(2, 2))


if __name__ == "__main__":
    unittest.main()
