"""Small importable dataset/task implementations using PyHealth's split interface."""

import numpy as np

from pyhealth.datasets import SampleBuilder, SubjectSplit, create_sample_dataset
from pyhealth.tasks import BaseTask


class SyntheticTask(BaseTask):
    """Preserve raw samples and choose the label processor through output_schema.

    mode="multiclass" expects one class label per sample. mode="multilabel"
    expects a list of labels and encodes it as a binary class-membership vector.
    This task does not convert an incorrectly labelled dataset into another task.
    """

    task_name = "synthetic_classification"
    input_schema = {"x": "tensor"}

    def __init__(self, mode="multiclass"):
        super().__init__()
        self.output_schema = {"label": mode}

    def __call__(self, sample):
        return [sample]


class SyntheticDataset:
    """In-memory source exposing the same set_task interface as BaseDataset.

    mode="multiclass" generates integer labels, e.g. 2; mode="multilabel"
    generates label lists, e.g. [0, 2]. This changes the actual generated samples.
    mode is this constructor's argument, not a general CPBench dataset setting.
    seed controls feature noise, independently of the split and evaluation seeds.

    Role prefixes make explicit SubjectSplit examples convenient. PatientSplit
    can instead assign any patients to parts using ratios and its own seed;
    it ignores those prefixes and the original generated group boundaries.
    """

    def __init__(
        self,
        seed=0,
        n_train=60,
        n_calibration=60,
        n_test=30,
        mode="multiclass",
        uniform=False,
        test_class=None,
    ):
        rng = np.random.default_rng(seed)
        self.samples = []
        for role, count in (
            ("train", n_train),
            ("calibration", n_calibration),
            ("test", n_test),
        ):
            for i in range(count):
                label = i % 3 if role != "test" or test_class is None else test_class
                if mode == "multilabel":
                    label = [i % 3] if i % 2 == 0 else [i % 3, (i + 1) % 3]
                x = np.zeros(3) if uniform else rng.normal(size=3)
                if not uniform:
                    x[label] += 2
                self.samples.append(
                    {
                        "patient_id": f"{role}-{i}",
                        "record_id": f"{role}-{i}",
                        "x": x.tolist(),
                        "label": label,
                    }
                )

    def set_task(self, task, *, split, input_processors=None, output_processors=None):
        samples = [row for sample in self.samples for row in task(sample)]
        builder = SampleBuilder(
            task.input_schema,
            task.output_schema,
            input_processors=input_processors,
            output_processors=output_processors,
        )
        builder.fit(samples, split=split)
        return tuple(
            create_sample_dataset(
                [samples[int(i)] for i in indices],
                task.input_schema,
                task.output_schema,
                input_processors=builder.input_processors,
                output_processors=builder.output_processors,
            )
            for indices in builder.split_indices
        )


def make_splits(
    seed=0,
    n_train=60,
    n_calibration=60,
    n_test=30,
    uniform=False,
    test_class=None,
    mode="multiclass",
):
    """Native-call reference data for the synthetic examples and tests."""
    source = SyntheticDataset(
        seed=seed,
        n_train=n_train,
        n_calibration=n_calibration,
        n_test=n_test,
        uniform=uniform,
        test_class=test_class,
        mode=mode,
    )
    roles = ("train", "calibration", "test")
    split = SubjectSplit(
        subjects=[
            [f"{role}-{i}" for i in range(count)]
            for role, count in zip(roles, (n_train, n_calibration, n_test))
        ]
    )
    return dict(zip(roles, source.set_task(SyntheticTask(mode), split=split)))
