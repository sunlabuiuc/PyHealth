"""Task views: several input schemas over the same samples, labels and split."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import BaseDataset, PatientSplit, SplitPlan
from pyhealth.models import BaseModel, RNN_V2
from pyhealth.processors.base_processor import FeatureProcessor
from pyhealth.tasks import BaseTask, YAIBMortalityMIMIC4
from pyhealth.tasks.survival_mimic4 import InHospitalSurvivalMIMIC4
from pyhealth.tasks.views import YAIB_PREPROCESSING, numeric_series_views

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
    feature_roles = {"value": "vitals_labs"}

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

        task = Plain()
        self.assertIsNone(task.view)
        self.assertEqual(
            task.cache_key(),
            {"input_schema": {"x": "raw"}, "output_schema": {"y": "binary"}},
        )
        with self.assertRaisesRegex(ValueError, "no views"):
            Plain(view="dense")

    def test_unknown_default_view_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "default_view"):

            class Bad(BaseTask):
                task_name = "bad"
                views = {"a": {"x": "raw"}}
                default_view = "b"
                output_schema = {"y": "binary"}

                def __call__(self, patient):
                    return []

    def test_subclass_with_its_own_schema_drops_inherited_views(self):
        # InHospitalSurvivalMIMIC4 extends the YAIB base with other fields.
        self.assertEqual(InHospitalSurvivalMIMIC4.views, {})
        self.assertIsNone(InHospitalSurvivalMIMIC4().view)


class TestCacheKey(unittest.TestCase):
    def test_views_share_a_key(self):
        self.assertEqual(ViewTask().cache_key(), ViewTask(view="scaled").cache_key())
        self.assertEqual(
            ViewTask().cache_key(), ViewTask(view={"value": "raw"}).cache_key()
        )

    def test_arguments_and_constants_change_the_key(self):
        self.assertNotEqual(ViewTask().cache_key(), ViewTask(keep_odd=False).cache_key())

        class Stricter(ViewTask):
            THRESHOLD = 1

        self.assertNotEqual(ViewTask().cache_key(), Stricter().cache_key())
        self.assertEqual(Stricter().cache_key()["class_constants"]["THRESHOLD"], 1)


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

    def test_recipe_returns_independent_schemas(self):
        first = numeric_series_views(dynamic=3, static=2)
        first["jagged"]["dynamic"][1]["levels"][0]["impute"].clear()
        second = numeric_series_views(dynamic=3, static=2)
        self.assertEqual(
            len(second["jagged"]["dynamic"][1]["levels"][0]["impute"]),
            len(YAIB_PREPROCESSING.series_impute),
        )

    def test_view_for_model_and_wards_argument(self):
        task = YAIBMortalityMIMIC4(icu_wards=["X"], model=RNN_V2)
        self.assertEqual(task.view, "jagged")
        self.assertEqual(task.icu_wards, ["X"])
        self.assertEqual(YAIBMortalityMIMIC4.feature_roles["dynamic"], "vitals_labs")


class TestSplitPlan(unittest.TestCase):
    IDS = [f"p{i:03d}" for i in range(N_PATIENTS)]

    def test_draw_assigns_every_patient_once(self):
        plan = SplitPlan.draw("demo", self.IDS, PatientSplit((0.6, 0.2, 0.2), seed=1))
        self.assertEqual(list(plan.parts), ["train", "val", "test"])
        listed = [pid for part in plan.parts.values() for pid in part]
        self.assertEqual(sorted(listed), self.IDS)

    def test_a_patient_keeps_its_part_whatever_the_cohort(self):
        plan = SplitPlan.draw("demo", self.IDS, PatientSplit((0.6, 0.2, 0.2), seed=1))
        part_of = {pid: name for name, part in plan.parts.items() for pid in part}
        for cohort in (self.IDS, self.IDS[::2], self.IDS[5:17]):
            mapping = {pid: [k] for k, pid in enumerate(cohort)}
            for name, indices in zip(plan.parts, plan.split_indices(mapping)):
                for k in indices:
                    self.assertEqual(part_of[cohort[int(k)]], name)

    def test_save_and_load_round_trip(self):
        plan = SplitPlan.draw("demo", self.IDS, PatientSplit((0.7, 0.3), seed=4))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "splits" / "demo.json"
            plan.save(path)
            loaded = SplitPlan.load(path)
        self.assertEqual(loaded, plan)
        self.assertEqual(loaded.to_dict(), plan.to_dict())
        self.assertEqual(list(loaded.parts), ["train", "test"])

    def test_overlapping_parts_are_rejected(self):
        with self.assertRaises(ValueError):
            SplitPlan("bad", {"train": ["a", "b"], "test": ["b"]})


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
        cls.plan = SplitPlan.load_or_draw(
            cls.dataset, "seed0", PatientSplit((0.6, 0.2, 0.2), seed=0)
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _set(self, task):
        return self.dataset.set_task(task, split=self.plan, num_workers=1)

    def test_plan_is_saved_and_reloaded(self):
        path = Path(self.dataset.cache_dir) / "splits" / "seed0.json"
        self.assertTrue(path.exists())
        again = SplitPlan.load_or_draw(
            self.dataset, "seed0", PatientSplit((0.5, 0.5), seed=9)
        )
        self.assertEqual(again, self.plan)

    def test_views_share_samples_labels_and_split(self):
        plain = self._set(ViewTask())
        scaled = self._set(ViewTask(view="scaled"))
        for part_plain, part_scaled, name in zip(plain, scaled, self.plan.parts):
            rows_plain = [
                (part_plain[i]["patient_id"], int(part_plain[i]["label"]))
                for i in range(len(part_plain))
            ]
            rows_scaled = [
                (part_scaled[i]["patient_id"], int(part_scaled[i]["label"]))
                for i in range(len(part_scaled))
            ]
            self.assertEqual(rows_plain, rows_scaled)
            self.assertTrue({pid for pid, _ in rows_plain} <= set(self.plan.parts[name]))
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
            1.0 + int(pid[1:]) + int(pid[1:]) % 2 for pid in self.plan.parts["train"]
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


if __name__ == "__main__":
    unittest.main()
