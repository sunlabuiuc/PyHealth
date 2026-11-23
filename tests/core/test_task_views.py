"""Task views: several input schemas over the same samples, labels and split."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import pyhealth.models as models
from pyhealth.datasets import (
    BaseDataset,
    PatientSplit,
    SubjectSplit,
    create_sample_dataset,
    get_dataloader,
)
from pyhealth.models import BaseModel, RNN, RNN_V2
from pyhealth.processors.base_processor import FeatureProcessor
from pyhealth.tasks import BaseTask, YAIBMortalityMIMIC4
from pyhealth.tasks.survival_mimic4 import InHospitalSurvivalMIMIC4

N_PATIENTS = 30

CONFIG = """version: "1.0"
tables:
  events:
    file_path: "events.csv"
    patient_id: "patient_id"
    timestamp: "time"
    attributes:
      - "value"
"""

CALLS = {"count": 0}


class Scale(FeatureProcessor):
    """Divides by the largest value seen in fit (a learned statistic)."""

    learns_statistics = True
    stores_tensor = True

    def __init__(self, offset: float = 0.0):
        self.offset = offset
        self.max = None

    def fit(self, samples, field):
        self.max = max(float(s[field][0]) for s in samples)

    def process(self, value):
        return torch.tensor([float(value[0]) / self.max + self.offset])


class ViewTask(BaseTask):
    task_name = "view_task"
    views = {
        "plain": {"value": "raw"},
        "scaled": {"value": Scale},
        "shifted": {"value": (Scale, {"offset": 5.0})},
    }
    default_view = "plain"
    output_schema = {"label": "binary"}

    THRESHOLD = 0

    def __init__(self, keep_odd: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.keep_odd = keep_odd

    def __call__(self, patient):
        index = int(patient.patient_id[1:])
        if not self.keep_odd and index % 2:
            return []
        return [
            {
                "patient_id": patient.patient_id,
                "value": [float(event.value)],
                "label": (index // 2) % 2,
            }
            for event in patient.get_events("events")
        ]


class ScaledModel(BaseModel):
    accepted_views = ("dense", "scaled")


class UndeclaredModel(BaseModel):
    pass


class JaggedOnlyModel(BaseModel):
    accepted_views = ("jagged",)


def _values(ds):
    return {ds[i]["patient_id"]: float(ds[i]["value"][0]) for i in range(len(ds))}


class TestViewSelection(unittest.TestCase):
    def test_default_view_is_the_class_schema(self):
        self.assertEqual(ViewTask.input_schema, {"value": "raw"})
        task = ViewTask()
        self.assertEqual(task.view, "plain")
        self.assertEqual(task.input_schema, {"value": "raw"})

    def test_view_by_name(self):
        task = ViewTask(view="scaled")
        self.assertEqual(task.view, "scaled")
        self.assertEqual(task.input_schema, {"value": Scale})
        # Selecting a view on one instance leaves the class untouched.
        self.assertEqual(ViewTask().view, "plain")

    def test_view_for_model_takes_first_accepted_view_offered(self):
        self.assertEqual(ViewTask(model=ScaledModel).view, "scaled")

    def test_custom_schema(self):
        task = ViewTask(view={"value": (Scale, {"offset": 1.0})})
        self.assertEqual(task.view, "custom")
        self.assertEqual(task.input_schema["value"][1], {"offset": 1.0})

    def test_errors_name_what_is_offered(self):
        with self.assertRaisesRegex(ValueError, "no view 'dense'.*'plain', 'scaled'"):
            ViewTask(view="dense")
        with self.assertRaisesRegex(ValueError, "does not declare accepted_views"):
            ViewTask(model=UndeclaredModel)
        with self.assertRaisesRegex(ValueError, "No view fits both"):
            ViewTask(model=JaggedOnlyModel)
        with self.assertRaisesRegex(ValueError, "not both"):
            ViewTask(view="scaled", model=ScaledModel)

    def test_task_without_views_is_unchanged(self):
        class Plain(BaseTask):
            task_name = "plain"
            input_schema = {"x": "raw"}
            output_schema = {"y": "binary"}

            def __call__(self, patient):
                return []

        self.assertIsNone(Plain().view)
        with self.assertRaisesRegex(ValueError, "no views"):
            Plain(view="dense")
        with self.assertRaisesRegex(ValueError, "no views"):
            Plain(view_overrides={"x": "tensor"})

    def test_unknown_default_view_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "default_view"):

            class Bad(BaseTask):
                task_name = "bad"
                views = {"a": {"x": "raw"}}
                default_view = "b"
                output_schema = {"y": "binary"}

                def __call__(self, patient):
                    return []

    def test_subclass_can_change_only_the_default_view(self):
        class DenseFirst(ViewTask):
            default_view = "scaled"

        self.assertEqual(DenseFirst().view, "scaled")
        self.assertEqual(DenseFirst.input_schema, {"value": Scale})
        self.assertEqual(ViewTask().view, "plain")
        with self.assertRaisesRegex(TypeError, "not one of its views"):

            class Wrong(ViewTask):
                default_view = "missing"

        with self.assertRaisesRegex(TypeError, "has no views"):

            class NoViews(BaseTask):
                task_name = "no_views"
                input_schema = {"x": "raw"}
                output_schema = {"y": "binary"}
                default_view = "x"

                def __call__(self, patient):
                    return []

    def test_subclass_with_its_own_schema_drops_inherited_views(self):
        # InHospitalSurvivalMIMIC4 extends the YAIB base with other fields.
        self.assertEqual(InHospitalSurvivalMIMIC4.views, {})
        self.assertIsNone(InHospitalSurvivalMIMIC4().view)


class TestYAIBViews(unittest.TestCase):
    def test_default_view_is_the_yaib_preprocessing(self):
        task = YAIBMortalityMIMIC4()
        self.assertEqual(task.view, "jagged")
        name, config = task.input_schema["dynamic"]
        self.assertEqual(name, "jagged")
        self.assertEqual(
            config["levels"][0],
            {
                "ordering": "series",
                "values": [len(YAIBMortalityMIMIC4.DYNAMIC_FEATURES)],
                "impute": [
                    {"method": "forward_fill"},
                    {"method": "mean", "by_dims": ()},
                    {"method": "zero"},
                ],
                "normalize": {"method": "zscore"},
                "missing_indicator": True,
            },
        )
        self.assertEqual(
            task.input_schema["static"][1]["levels"][0],
            {
                "ordering": "single",
                "values": [len(YAIBMortalityMIMIC4.STATIC_FEATURES)],
                "impute": [{"method": "mean", "by_dims": ()}, {"method": "zero"}],
                "normalize": {"method": "zscore"},
                "missing_indicator": True,
            },
        )

    def test_the_three_datasets_define_the_same_views(self):
        """Each task file is self-contained, so the copies must not drift."""
        from pyhealth.tasks import YAIBMortalityEICU, YAIBMortalityMIMIC3

        for other in (YAIBMortalityMIMIC3, YAIBMortalityEICU):
            self.assertEqual(sorted(other.views), sorted(YAIBMortalityMIMIC4.views))
            for view in YAIBMortalityMIMIC4.views:
                ours, theirs = YAIBMortalityMIMIC4.views[view], other.views[view]
                for key in ours:
                    self.assertEqual(ours[key][0], theirs[key][0])
                    self.assertEqual(
                        _without_widths(ours[key][1]), _without_widths(theirs[key][1])
                    )

    def test_view_overrides_change_settings_of_one_field(self):
        task = YAIBMortalityMIMIC4(
            view="dense", view_overrides={"dynamic": {"normalize": False}}
        )
        self.assertEqual(task.view, "dense")
        self.assertEqual(
            task.input_schema["dynamic"],
            ("dense_series", {"forward_fill": True, "normalize": False, "missing_indicator": True}),
        )
        # Other fields and the class's own views are untouched.
        self.assertEqual(task.input_schema["static"], YAIBMortalityMIMIC4.views["dense"]["static"])
        self.assertTrue(YAIBMortalityMIMIC4.views["dense"]["dynamic"][1]["normalize"])

    def test_view_overrides_can_replace_a_processor_and_reject_unknown_keys(self):
        task = YAIBMortalityMIMIC4(view="dense", view_overrides={"static": "raw"})
        self.assertEqual(task.input_schema["static"], "raw")
        with self.assertRaisesRegex(ValueError, "not inputs of the view 'dense'"):
            YAIBMortalityMIMIC4(view="dense", view_overrides={"labs": {}})

    def test_subclass_can_add_a_named_view(self):
        class Mine(YAIBMortalityMIMIC4):
            views = {
                **YAIBMortalityMIMIC4.views,
                "unscaled": {
                    "dynamic": ("dense_series", {"normalize": False}),
                    "static": ("dense_static", {"normalize": False}),
                },
            }

        self.assertEqual(Mine().view, "jagged")
        self.assertEqual(Mine(view="unscaled").input_schema["static"][1], {"normalize": False})

    def test_view_for_model_and_wards_argument(self):
        task = YAIBMortalityMIMIC4(icu_wards=["X"], model=RNN_V2)
        self.assertEqual(task.view, "jagged")
        self.assertEqual(task.icu_wards, ["X"])


class TestSetTaskWithViews(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name) / "data"
        root.mkdir()
        rows = ["patient_id,time,value"]
        for k in range(N_PATIENTS):
            for v in range(1 + k % 2):
                rows.append(f"p{k:03d},2020-01-{1 + v:02d} 00:00:00,{1.0 + k + v}")
        (root / "events.csv").write_text("\n".join(rows) + "\n")
        (root / "config.yaml").write_text(CONFIG)
        cls.dataset = BaseDataset(
            root=str(root),
            tables=["events"],
            dataset_name="ViewDataset",
            config_path=str(root / "config.yaml"),
            cache_dir=str(Path(cls.tmp.name) / "cache"),
            num_workers=1,
        )
        cls.plan = SubjectSplit.load_or_draw(
            cls.dataset, "seed0", PatientSplit((0.6, 0.2, 0.2), seed=0)
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _set(self, task):
        return self.dataset.set_task(task, split=self.plan, num_workers=1)

    def test_split_is_saved_and_reloaded(self):
        path = Path(self.dataset.cache_dir) / "splits" / "seed0.json"
        self.assertTrue(path.exists())
        again = SubjectSplit.load_or_draw(
            self.dataset, "seed0", PatientSplit((0.5, 0.5), seed=9)
        )
        self.assertEqual(again, self.plan)

    def test_views_share_samples_labels_and_split(self):
        plain = self._set(ViewTask())
        scaled = self._set(ViewTask(view="scaled"))
        for part_plain, part_scaled, patients in zip(plain, scaled, self.plan.subjects):
            rows_plain = [
                (part_plain[i]["patient_id"], int(part_plain[i]["label"]))
                for i in range(len(part_plain))
            ]
            rows_scaled = [
                (part_scaled[i]["patient_id"], int(part_scaled[i]["label"]))
                for i in range(len(part_scaled))
            ]
            self.assertEqual(rows_plain, rows_scaled)
            self.assertTrue({pid for pid, _ in rows_plain} <= set(patients))
        self.assertEqual(scaled[0].view, "scaled")
        self.assertEqual(plain[0].view, "plain")

    def test_views_share_one_task_cache(self):
        self._set(ViewTask())
        self._set(ViewTask(view="scaled"))
        self._set(ViewTask(view="shifted"))
        task_dirs = list((Path(self.dataset.cache_dir) / "tasks").glob("view_task_*"))
        self.assertEqual(len(task_dirs), 1)
        # One processed cache per view (other tests may have added more).
        built = list(task_dirs[0].glob("samples_*.ld/index.json"))
        self.assertGreaterEqual(len(built), 3)
        # A different cohort is another task cache.
        self._set(ViewTask(keep_odd=False))
        task_dirs = list((Path(self.dataset.cache_dir) / "tasks").glob("view_task_*"))
        self.assertEqual(len(task_dirs), 2)

    def test_statistics_come_from_the_training_part(self):
        train, val, test = self._set(ViewTask(view="scaled"))
        self.assertTrue(np.isclose(max(_values(train).values()), 1.0))
        processor = train.input_processors["value"]
        train_raw = max(
            1.0 + int(pid[1:]) + int(pid[1:]) % 2 for pid in self.plan.subjects[0]
        )
        self.assertAlmostEqual(processor.max, train_raw)

    def test_fitted_view_needs_a_split(self):
        with self.assertRaisesRegex(ValueError, "Pass a split"):
            self.dataset.set_task(ViewTask(view="scaled"), num_workers=1)
        # A view that learns nothing does not.
        samples = self.dataset.set_task(ViewTask(), num_workers=1)
        self.assertEqual(samples.view, "plain")

    def test_model_rejects_a_view_it_does_not_read(self):
        train, _, _ = self._set(ViewTask())
        with self.assertRaisesRegex(ValueError, "built with the view 'plain'"):
            ScaledModel(train)
        UndeclaredModel(train)
        train, _, _ = self._set(ViewTask(model=ScaledModel))
        ScaledModel(train)


def _without_widths(settings):
    """Processor settings without the feature counts, which differ by dataset."""
    if isinstance(settings, dict):
        return {
            k: _without_widths(v) for k, v in settings.items() if k != "values"
        }
    if isinstance(settings, (list, tuple)):
        return [_without_widths(v) for v in settings]
    return settings


DENSE_MODELS = [
    "RNN", "MultimodalRNN", "TCN", "Transformer", "RETAIN", "MultimodalRETAIN",
    "MultimodalAdaCare", "GRASP", "Agent", "JambaEHR", "StageNet", "MLP",
    "LogisticRegression", "KnowRare", "BEHRT", "BiteNet", "UDC",
]


def _yaib_samples(n=16):
    """Synthetic YAIB-shaped stays: mostly-missing hourly values and statics."""
    rng = np.random.default_rng(0)
    n_dynamic = len(YAIBMortalityMIMIC4.DYNAMIC_FEATURES)
    n_static = len(YAIBMortalityMIMIC4.STATIC_FEATURES)
    samples = []
    for i in range(n):
        hours = int(rng.integers(5, 25))
        dynamic = rng.normal(size=(hours, n_dynamic))
        dynamic[rng.random((hours, n_dynamic)) < 0.6] = np.nan
        dynamic[:, 7] = np.nan  # a feature never measured
        static = rng.normal(size=(1, n_static))
        if i % 3 == 0:
            static[0, i % n_static] = np.nan
        samples.append(
            {
                "patient_id": f"p{i}",
                "dynamic": {"time": list(range(hours)), "values": dynamic.tolist()},
                "static": {"values": static.tolist()},
                "mortality": i % 2,
            }
        )
    return samples


class TestDenseView(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = _yaib_samples()
        cls.n_dynamic = len(YAIBMortalityMIMIC4.DYNAMIC_FEATURES)
        cls.dense = cls._dataset("dense")
        cls.jagged = cls._dataset("jagged")

    @classmethod
    def _dataset(cls, view):
        task = YAIBMortalityMIMIC4(view=view)
        return create_sample_dataset(cls.samples, task.input_schema, task.output_schema)

    def test_dense_values_equal_the_jagged_view(self):
        """Both views apply the same imputation and scaling to the same data."""
        f = self.n_dynamic
        for i in range(len(self.samples)):
            value, mask = self.dense[i]["dynamic"]
            jagged_value, jagged_missing, _ = self.jagged[i]["dynamic"]
            self.assertEqual(value.shape, (jagged_value.shape[0], 2 * f))
            self.assertTrue(torch.allclose(value[:, :f], jagged_value.float(), atol=1e-5))
            self.assertTrue(torch.equal(value[:, f:].bool(), jagged_missing))
            self.assertTrue(torch.equal(mask, torch.ones(value.shape[0])))
            static = self.dense[i]["static"]
            jagged_static = self.jagged[i]["static"][0].float().reshape(-1)
            self.assertTrue(torch.allclose(static[: len(jagged_static)], jagged_static, atol=1e-5))

    def test_forward_fill_then_training_mean(self):
        nan = float("nan")
        from pyhealth.processors import DenseSeriesProcessor

        processor = DenseSeriesProcessor(missing_indicator=False)
        train = [{"x": {"time": [0, 1], "values": [[1.0, nan], [3.0, nan]]}}]
        processor.fit(train, "x")
        value, mask = processor.process(
            {"time": [0, 1, 2], "values": [[nan, 5.0], [3.0, nan], [nan, nan]]}
        )
        # Feature 0: mean 2, std 1. Before its first measurement it is the
        # mean (0); afterwards the last measurement is carried forward.
        self.assertEqual(value[:, 0].tolist(), [0.0, 1.0, 1.0])
        # Feature 1 was never measured in training: left unscaled, then carried.
        self.assertEqual(value[:, 1].tolist(), [5.0, 5.0, 5.0])
        self.assertEqual(mask.tolist(), [1.0, 1.0, 1.0])

    def test_padding_does_not_change_a_prediction(self):
        """The explicit mask, not zeros in the data, marks padded steps."""
        alone = next(iter(get_dataloader(self.dense.subset([0]), batch_size=1)))
        batch = next(iter(get_dataloader(self.dense, batch_size=len(self.samples))))
        self.assertLess(alone["dynamic"][0].shape[1], batch["dynamic"][0].shape[1])
        # RNN pools by the last unmasked step, Transformer by attention.
        for name in ("RNN", "Transformer"):
            with self.subTest(model=name):
                torch.manual_seed(0)
                model = getattr(models, name)(self.dense).eval()
                with torch.no_grad():
                    difference = model(**alone)["logit"][0] - model(**batch)["logit"][0]
                self.assertLess(float(difference.abs().max()), 1e-5)

    def test_models_that_accept_dense_train_on_it(self):
        batch = next(iter(get_dataloader(self.dense, batch_size=8)))
        for name in DENSE_MODELS:
            with self.subTest(model=name):
                model_class = getattr(models, name)
                self.assertEqual(YAIBMortalityMIMIC4(model=model_class).view, "dense")
                torch.manual_seed(0)
                output = model_class(self.dense)(**batch)
                output["loss"].backward()
                self.assertTrue(bool(torch.isfinite(output["loss"])))

    def test_declared_models_are_the_tested_ones(self):
        declared = {
            name
            for name, obj in vars(models).items()
            if isinstance(obj, type)
            and issubclass(obj, BaseModel)
            and "dense" in obj.accepted_views
        }
        self.assertEqual(declared, set(DENSE_MODELS))


class TestEventsView(unittest.TestCase):
    """The foundation-model view: the same measurements as coded events."""

    @classmethod
    def setUpClass(cls):
        cls.samples = _yaib_samples(8)
        cls.task = YAIBMortalityMIMIC4(view="events")
        cls.dataset = create_sample_dataset(
            cls.samples, cls.task.input_schema, cls.task.output_schema
        )

    def test_one_event_per_measured_value(self):
        from pyhealth.processors.event_processor import ORIGIN

        names = YAIBMortalityMIMIC4.DYNAMIC_FEATURES
        for i, sample in enumerate(self.samples):
            values = np.array(sample["dynamic"]["values"])
            events = self.dataset[i]["dynamic"]
            self.assertEqual(len(events), int((~np.isnan(values)).sum()))
            rebuilt = np.full(values.shape, np.nan)
            for when, code, value, _ in events:
                hour = round((when - ORIGIN).total_seconds() / 3600)
                rebuilt[hour, names.index(code.split("/", 1)[1])] = value
            self.assertTrue(np.allclose(rebuilt, values, equal_nan=True))

    def test_birth_time_comes_from_the_age(self):
        from pyhealth.processors.event_processor import ORIGIN

        age = self.samples[1]["static"]["values"][0][0]
        birth = self.dataset[1]["static"]
        self.assertAlmostEqual((ORIGIN - birth).days / 365.25, age, places=2)

    def test_codes_can_be_mapped_to_a_model_vocabulary(self):
        from pyhealth.processors import get_processor

        task = YAIBMortalityMIMIC4(
            view="events",
            view_overrides={
                "dynamic": {"codes": {"crea": "LOINC/2160-0"}, "units": {"crea": "mg/dL"}}
            },
        )
        name, config = task.input_schema["dynamic"]
        processor = get_processor(name)(**config)
        names = YAIBMortalityMIMIC4.DYNAMIC_FEATURES
        row = [float("nan")] * len(names)
        row[names.index("crea")], row[names.index("hr")] = 1.2, 70.0
        events = processor.process({"time": [1], "values": [row]})
        self.assertEqual(
            sorted(e[1:] for e in events),
            sorted([("YAIB/hr", 70.0, None), ("LOINC/2160-0", 1.2, "mg/dL")]),
        )

    def test_foundation_model_trains_on_the_events_view(self):
        try:
            from transformers import GPT2Config, GPT2Model
        except ImportError:
            self.skipTest("transformers is not installed")
        from pyhealth.models import CLMBR, HFEHRFoundationModel
        from pyhealth.models.ehr_fm import HFEHRTokenizer

        self.assertEqual(YAIBMortalityMIMIC4(model=HFEHRFoundationModel).view, "events")
        self.assertEqual(YAIBMortalityMIMIC4(model=CLMBR).view, "events")
        tokens = []
        for name in YAIBMortalityMIMIC4.DYNAMIC_FEATURES:
            for low, high in ((-1e308, 0.0), (0.0, 1e308)):
                tokens.append({
                    "code": f"YAIB/{name}", "type": "numerical_range",
                    "tokenization": {"unit": "None", "range_start": low, "range_end": high},
                })
        tokenizer = HFEHRTokenizer({"tokens": tokens})
        backbone = GPT2Model(GPT2Config(
            vocab_size=len(tokenizer), n_positions=64, n_embd=8, n_layer=1, n_head=2,
        ))
        model = HFEHRFoundationModel(
            self.dataset, backbone=backbone, tokenizer=tokenizer, max_length=64
        )
        # The model finds the view's fields without being told their keys.
        self.assertEqual((model.events_key, model.birth_key), ("dynamic", "static"))
        batch = next(iter(get_dataloader(self.dataset, batch_size=4)))
        output = model(**batch)
        self.assertTrue(bool(torch.isfinite(output["loss"])))
        self.assertEqual(model.coverage_report()["coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
