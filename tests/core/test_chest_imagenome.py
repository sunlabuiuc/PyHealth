"""ChestImaGenomeDataset, using fabricated scene graphs and gold comparisons."""

import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pyhealth.datasets import ChestImaGenomeDataset
from pyhealth.datasets.chest_imagenome import (
    COMPARISONS_CSV,
    GOLD_COMPARISONS_CSV,
    IMAGES_CSV,
    OBJECTS_CSV,
    prepare_chest_imagenome,
)

GOLD_HEADER = (
    "patient_id",
    "current_image_id",
    "previous_image_id",
    "bbox",
    "label_name",
    "comparison",
)


def box(name: str, x1: int = 10, y1: int = 20) -> dict:
    return {
        "object_id": f"ignored_{name}",
        "bbox_name": name,
        "x1": 1,
        "original_x1": x1,
        "original_y1": y1,
        "original_x2": x1 + 5,
        "original_y2": y1 + 5,
    }


def relationship(
    current: str,
    previous: str,
    region: str = "left lung",
    names: tuple[str, ...] = ("comparison|yes|worsened",),
    attributes: tuple[str, ...] = ("anatomicalfinding|yes|lung opacity",),
) -> dict:
    return {
        "subject_id": f"{current}_{region}",
        "object_id": f"{previous}_{region}",
        "bbox_name": region,
        "relationship_names": list(names),
        "attributes": list(attributes),
    }


def write_graph(root: Path, dicom_id: str, patient: int, study: int, **fields) -> None:
    folder = root / "silver_dataset" / "scene_graph"
    folder.mkdir(parents=True, exist_ok=True)
    graph = {
        "patient_id": patient,
        "study_id": study,
        "StudyDateTime": "2150-01-02 03:04:05",
        "StudyOrder": 1,
        "viewpoint": "AP",
        "gender": "F",
        "age_decile": "60-70",
        "objects": [],
        "relationships": [],
        **fields,
    }
    (folder / f"{dicom_id}_SceneGraph.json").write_text(json.dumps(graph))


def write_gold(root: Path, rows: list[tuple], header: tuple = GOLD_HEADER) -> None:
    folder = root / "gold_dataset"
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(header)] + ["\t".join(row) for row in rows]
    path = folder / "gold_object_comparison_with_coordinates.txt"
    path.write_text("\n".join(lines) + "\n")


def write_fixture(root: Path) -> None:
    """Patient 1 has a pair of images, patient 2 one image with no comparison."""
    write_graph(root, "img-a", 1, 100, objects=[box("left lung")])
    write_graph(
        root,
        "img-b",
        1,
        101,
        StudyDateTime="2150-02-02 03:04:05",
        StudyOrder=2,
        objects=[box("left lung"), box("right lung")],
        relationships=[
            relationship(
                "img-b",
                "img-a",
                attributes=(
                    "anatomicalfinding|yes|lung opacity",
                    "disease|no|pneumonia",
                    "technicalassessment|yes|low lung volumes",
                ),
            ),
            relationship(
                "img-b",
                "img-a",
                region="right lung",
                names=("comparison|yes|worsened", "comparison|yes|improved"),
            ),
            relationship(
                "img-b", "img-a", region="trachea", names=("comparison|yes|other",)
            ),
        ],
    )
    write_graph(root, "img-c", 2, 200)
    write_gold(
        root,
        [
            ("1", "img-b", "img-a", "left lung", "lung opacity", "improved"),
            ("1", "img-b", "img-a", "left lung", "lung opacity", "improved"),
            ("3", "img-x", "img-y", "left lung", "atelectasis", "no change"),
        ],
    )


def read_table(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(root / name, dtype=str, keep_default_na=False)


class TestPrepareChestImaGenome(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        write_fixture(self.root)

    def test_images_keep_mimic_cxr_ids(self):
        prepare_chest_imagenome(self.root)
        images = read_table(self.root, IMAGES_CSV).set_index("dicom_id")
        self.assertEqual(sorted(images.index), ["img-a", "img-b", "img-c"])
        self.assertEqual(images.loc["img-b", "subject_id"], "1")
        self.assertEqual(images.loc["img-b", "study_id"], "101")
        self.assertEqual(images.loc["img-b", "study_order"], "2")
        self.assertEqual(images.loc["img-b", "study_datetime"], "2150-02-02 03:04:05")

    def test_comparison_links_current_and_previous_study(self):
        prepare_chest_imagenome(self.root)
        comparisons = read_table(self.root, COMPARISONS_CSV)
        row = comparisons[comparisons["bbox_name"] == "left lung"].iloc[0]
        self.assertEqual(
            (row["dicom_id"], row["study_id"]), ("img-b", "101")
        )
        self.assertEqual(
            (row["previous_dicom_id"], row["previous_study_id"]), ("img-a", "100")
        )

    def test_comparison_keeps_diseases_and_their_polarity(self):
        prepare_chest_imagenome(self.root)
        comparisons = read_table(self.root, COMPARISONS_CSV)
        row = comparisons[comparisons["bbox_name"] == "left lung"].iloc[0]
        self.assertEqual(row["findings"], "lung opacity;pneumonia")
        self.assertEqual(row["finding_polarities"], "yes;no")
        self.assertEqual(row["comparisons"], "worsened")

    def test_ambiguous_comparison_is_kept_for_the_task(self):
        prepare_chest_imagenome(self.root)
        comparisons = read_table(self.root, COMPARISONS_CSV)
        row = comparisons[comparisons["bbox_name"] == "right lung"].iloc[0]
        self.assertEqual(row["comparisons"], "worsened;improved")

    def test_relationship_without_a_progression_label_is_dropped(self):
        prepare_chest_imagenome(self.root)
        comparisons = read_table(self.root, COMPARISONS_CSV)
        self.assertNotIn("trachea", set(comparisons["bbox_name"]))

    def test_previous_image_without_graph_leaves_study_blank(self):
        write_graph(
            self.root, "img-d", 2, 201, relationships=[relationship("img-d", "gone")]
        )
        prepare_chest_imagenome(self.root)
        comparisons = read_table(self.root, COMPARISONS_CSV)
        row = comparisons[comparisons["dicom_id"] == "img-d"].iloc[0]
        self.assertEqual(row["previous_dicom_id"], "gone")
        self.assertEqual(row["previous_study_id"], "")

    def test_objects_keep_original_coordinates(self):
        prepare_chest_imagenome(self.root)
        objects = read_table(self.root, OBJECTS_CSV)
        self.assertEqual(len(objects), 3)
        row = objects[objects["dicom_id"] == "img-a"].iloc[0]
        self.assertEqual(
            row[["original_x1", "original_y1", "original_x2", "original_y2"]].tolist(),
            ["10", "20", "15", "25"],
        )

    def test_object_corrections_match_dipro(self):
        renamed = "38bbebbd-b67908dc-387567bf-e3f965b7-77a18536"
        dropped = "3e953c67-28b78460-4f783437-379841d3-a79ec530"
        write_graph(
            self.root, renamed, 4, 400, objects=[box("right apical zone", 1146, 532)]
        )
        write_graph(
            self.root,
            dropped,
            5,
            500,
            objects=[
                box("left mid lung zone", 1500, 886),
                box("left mid lung zone", 200, 300),
            ],
        )
        raw = self.root / "silver_dataset/scene_graph" / f"{dropped}_SceneGraph.json"
        before = raw.read_text()
        prepare_chest_imagenome(self.root)
        objects = read_table(self.root, OBJECTS_CSV)
        self.assertEqual(
            objects[objects["dicom_id"] == renamed]["bbox_name"].tolist(),
            ["left apical zone"],
        )
        self.assertEqual(
            objects[objects["dicom_id"] == dropped]["original_x1"].tolist(), ["200"]
        )
        self.assertEqual(raw.read_text(), before)

    def test_gold_is_deduplicated_and_joined_to_studies(self):
        prepare_chest_imagenome(self.root)
        gold = read_table(self.root, GOLD_COMPARISONS_CSV)
        self.assertEqual(len(gold), 2)
        known, unknown = gold.iloc[0], gold.iloc[1]
        self.assertEqual(known["study_id"], "101")
        self.assertEqual(known["previous_study_id"], "100")
        self.assertEqual(known["study_datetime"], "2150-02-02 03:04:05")
        self.assertEqual(known["finding"], "lung opacity")
        self.assertEqual(known["comparison"], "improved")
        self.assertEqual(unknown["study_id"], "")
        self.assertEqual(unknown["study_datetime"], "")

    def test_gold_without_required_column_raises(self):
        write_gold(self.root, [], header=GOLD_HEADER[:-1])
        with self.assertRaisesRegex(ValueError, "comparison"):
            prepare_chest_imagenome(self.root)

    def test_unreadable_scene_graph_is_skipped(self):
        path = self.root / "silver_dataset/scene_graph/bad_SceneGraph.json"
        path.write_text("{not json")
        with self.assertLogs("pyhealth.datasets.chest_imagenome", logging.WARNING):
            prepare_chest_imagenome(self.root)
        self.assertNotIn("bad", set(read_table(self.root, IMAGES_CSV)["dicom_id"]))

    def test_datetime_with_time_zone_and_bad_datetime(self):
        write_graph(self.root, "img-e", 6, 600, StudyDateTime="2150-03-04 05:06:07 UTC")
        write_graph(self.root, "img-f", 6, 601, StudyDateTime="not a date")
        prepare_chest_imagenome(self.root)
        images = read_table(self.root, IMAGES_CSV).set_index("dicom_id")
        self.assertEqual(images.loc["img-e", "study_datetime"], "2150-03-04 05:06:07")
        self.assertEqual(images.loc["img-f", "study_datetime"], "")

    def test_unparseable_datetimes_are_reported(self):
        write_graph(self.root, "img-f", 6, 601, StudyDateTime="not a date")
        with self.assertLogs("pyhealth.datasets.chest_imagenome", logging.WARNING):
            prepare_chest_imagenome(self.root)

    def test_numeric_datetime_is_blank_and_reported(self):
        write_graph(self.root, "img-g", 6, 602, StudyDateTime=20150102030405)
        write_graph(self.root, "img-h", 6, 603, StudyDateTime=1.5e9)
        with self.assertLogs("pyhealth.datasets.chest_imagenome", logging.WARNING) as logs:
            prepare_chest_imagenome(self.root)
        images = read_table(self.root, IMAGES_CSV).set_index("dicom_id")
        self.assertEqual(images.loc["img-g", "study_datetime"], "")
        self.assertEqual(images.loc["img-h", "study_datetime"], "")
        self.assertTrue(any("no usable StudyDateTime" in line for line in logs.output))

    def test_scene_graph_without_ids_is_skipped(self):
        for dicom_id, missing in (("no-patient", "patient_id"), ("no-study", "study_id")):
            write_graph(self.root, dicom_id, 7, 700)
            path = self.root / f"silver_dataset/scene_graph/{dicom_id}_SceneGraph.json"
            graph = json.loads(path.read_text())
            del graph[missing]
            path.write_text(json.dumps(graph))
        with self.assertLogs("pyhealth.datasets.chest_imagenome", logging.WARNING) as logs:
            prepare_chest_imagenome(self.root)
        dicom_ids = set(read_table(self.root, IMAGES_CSV)["dicom_id"])
        self.assertNotIn("no-patient", dicom_ids)
        self.assertNotIn("no-study", dicom_ids)
        self.assertEqual(sum("missing or malformed" in line for line in logs.output), 2)

    def test_scene_graph_with_a_malformed_object_writes_nothing(self):
        write_graph(self.root, "bad-object", 7, 701, objects=[{"bbox_name": "left lung"}])
        with self.assertLogs("pyhealth.datasets.chest_imagenome", logging.WARNING):
            prepare_chest_imagenome(self.root)
        self.assertNotIn("bad-object", set(read_table(self.root, IMAGES_CSV)["dicom_id"]))
        self.assertNotIn(
            "bad-object", set(read_table(self.root, OBJECTS_CSV)["dicom_id"])
        )

    def test_files_nested_by_the_zip_are_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_graph(root, "img-a", 1, 100)
            flat = root / "silver_dataset" / "scene_graph"
            nested = flat / "scene_graph"
            nested.mkdir()
            (flat / "img-a_SceneGraph.json").rename(
                nested / "img-a_SceneGraph.json"
            )
            prepare_chest_imagenome(root)
            self.assertEqual(len(read_table(root, IMAGES_CSV)), 1)

    def test_read_only_root_writes_to_the_home_cache(self):
        with tempfile.TemporaryDirectory() as cache, patch(
            "pyhealth.datasets.chest_imagenome.os.access", return_value=False
        ), patch(
            "pyhealth.datasets.chest_imagenome.metadata_cache_dir",
            return_value=Path(cache) / "tables",
        ):
            out = prepare_chest_imagenome(self.root)
            self.assertEqual(out, Path(cache) / "tables")
            self.assertTrue((out / IMAGES_CSV).exists())
            self.assertFalse((self.root / IMAGES_CSV).exists())

    def test_missing_scene_graphs_raise(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(FileNotFoundError, "scene_graph.zip"),
        ):
            prepare_chest_imagenome(tmp)


class TestChestImaGenomeDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.tmp = Path(tmp.name)
        root = cls.tmp / "shared"
        root.mkdir()
        write_fixture(root)
        cls.dataset = ChestImaGenomeDataset(
            root=str(root), cache_dir=cls.tmp / "shared_cache"
        )

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(dir=self.tmp)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "data"
        self.root.mkdir()
        self.cache = Path(tmp.name) / "cache"
        write_fixture(self.root)

    def load(self, **kwargs) -> ChestImaGenomeDataset:
        return ChestImaGenomeDataset(
            root=str(self.root), cache_dir=self.cache, **kwargs
        )

    def test_patient_events_are_readable_by_a_task(self):
        patient = self.dataset.get_patient("1")
        self.assertEqual(len(patient.get_events("silver_images")), 2)
        self.assertEqual(len(patient.get_events("silver_objects")), 3)
        event = patient.get_events("silver_comparisons")[0]
        self.assertEqual(event["previous_study_id"], "100")
        self.assertEqual(event["timestamp"].year, 2150)
        self.assertEqual(event["timestamp"].month, 2)
        gold = patient.get_events("gold_comparisons")[0]
        self.assertEqual(gold["finding"], "lung opacity")

    def test_patient_without_comparisons_has_no_events(self):
        patient = self.dataset.get_patient("2")
        self.assertEqual(len(patient.get_events("silver_images")), 1)
        self.assertEqual(patient.get_events("silver_comparisons"), [])

    def test_gold_pair_without_scene_graph_loads_without_timestamp(self):
        event = self.dataset.get_patient("3").get_events("gold_comparisons")[0]
        self.assertEqual(event["dicom_id"], "img-x")
        self.assertIsNone(event["study_id"])

    def test_missing_gold_file_gives_an_empty_table(self):
        (self.root / "gold_dataset/gold_object_comparison_with_coordinates.txt").unlink()
        with self.assertLogs("pyhealth.datasets.chest_imagenome", logging.WARNING):
            dataset = self.load(tables=["gold_comparisons"])
        self.assertEqual(dataset.unique_patient_ids, [])

    def test_tables_can_be_chosen(self):
        dataset = self.load(tables=["silver_comparisons"])
        self.assertEqual(sorted(dataset.unique_patient_ids), ["1"])

    def test_existing_tables_are_reused(self):
        self.load()
        with patch(
            "pyhealth.datasets.chest_imagenome.prepare_chest_imagenome"
        ) as prepare:
            self.load()
        prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
