import contextlib
import io
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import narwhals as nw
import polars as pl

from pyhealth.datasets import (
    FHIRDataset,
    SyntheaDataset,
    SyntheaFHIRDataset,
    SyntheaGenerator,
)
from pyhealth.datasets.base_dataset import BaseDataset
from pyhealth.datasets.fhir.utils import flatten_resource
from pyhealth.datasets.synthea_csv import DEFAULT_TABLES


class TestSyntheaGenerator(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="synthea_")
        self.tmp = Path(self._tmp.name)
        self.output = self.tmp / "output"
        self.cache = self.tmp / "cache"

    def tearDown(self):
        self._tmp.cleanup()

    def generator(self, **kwargs):
        return SyntheaGenerator(
            output_dir=self.output,
            auto_download=False,
            **kwargs,
        )

    def properties_jar(self) -> Path:
        jar = self.tmp / "synthea.jar"
        with zipfile.ZipFile(jar, "w") as archive:
            archive.writestr(
                "synthea.properties",
                """\
generate.thread_pool_size = -1
generate.only_alive_patients = false
exporter.years_of_history = 10
exporter.baseDirectory = ./output
exporter.csv.export = false
exporter.csv.append_mode = false
exporter.csv.folder_per_run = false
exporter.fhir.export = true
exporter.fhir.bulk_data = false
exporter.fhir.use_us_core_ig = true
exporter.ccda.export = false
#exporter.code_map.icd10-cm=code-map.json
""",
            )
        return jar

    def test_generator_is_representation_agnostic(self):
        generator = self.generator(
            synthea_config={
                "exporter.csv.export": True,
                "exporter.fhir.export": True,
            }
        )

        self.assertNotIsInstance(generator, BaseDataset)
        self.assertTrue(generator.csv_enabled)
        self.assertTrue(generator.fhir_enabled)
        self.assertEqual(generator.output_path("csv"), generator.generation_dir / "csv")
        self.assertEqual(
            generator.output_path("fhir"), generator.generation_dir / "fhir"
        )

    def test_generation_configuration_partitions_output(self):
        first = self.generator(seed=1)
        repeated = self.generator(seed=1)
        second = self.generator(seed=2)
        csv = self.generator(
            seed=1,
            synthea_config={
                "exporter.csv.export": True,
                "exporter.fhir.export": False,
            },
        )
        configured = self.generator(
            seed=1,
            synthea_config={"generate.thread_pool_size": 4},
        )

        self.assertEqual(first.generation_dir, repeated.generation_dir)
        self.assertNotEqual(first.generation_dir, second.generation_dir)
        self.assertNotEqual(first.generation_dir, csv.generation_dir)
        self.assertNotEqual(first.generation_dir, configured.generation_dir)

    def test_constructor_validation(self):
        for population in (0, -1):
            with self.subTest(population=population), self.assertRaises(ValueError):
                self.generator(population=population)
        for population in (True, "10"):
            with self.subTest(population=population), self.assertRaises(TypeError):
                self.generator(population=population)
        with self.assertRaises(TypeError):
            self.generator(seed="42")
        with self.assertRaisesRegex(ValueError, "city requires state"):
            self.generator(city="Boston")
        with self.assertRaisesRegex(ValueError, "CSV, Bulk FHIR, or both"):
            self.generator(synthea_config={"exporter.fhir.export": False})

    def test_csv_argv_only_sets_required_exporter_values(self):
        generator = self.generator(
            population=25,
            seed=42,
            state="Massachusetts",
            city="Boston",
            synthea_config={
                "exporter.csv.export": True,
                "exporter.fhir.export": False,
                "generate.thread_pool_size": 4,
                "exporter.csv.append_mode": True,
            },
        )

        argv = generator.build_argv(Path("java"), Path("synthea.jar"))

        self.assertIn("--exporter.csv.export=true", argv)
        self.assertIn("--exporter.fhir.export=false", argv)
        self.assertNotIn("--exporter.fhir.bulk_data=false", argv)
        self.assertIn("--exporter.csv.append_mode=true", argv)
        self.assertNotIn("--exporter.csv.folder_per_run=false", argv)
        self.assertEqual(argv[argv.index("-p") + 1], "25")
        self.assertEqual(argv[argv.index("-s") + 1], "42")
        self.assertEqual(argv[-2:], ["Massachusetts", "Boston"])

    def test_bulk_fhir_and_csv_can_be_enabled_together(self):
        argv = self.generator(
            synthea_config={
                "exporter.csv.export": True,
                "exporter.fhir.export": True,
            }
        ).build_argv(Path("java"), Path("synthea.jar"))

        self.assertIn("--exporter.csv.export=true", argv)
        self.assertIn("--exporter.fhir.export=true", argv)
        self.assertIn("--exporter.fhir.bulk_data=true", argv)

    def test_synthea_config_selects_the_three_supported_output_modes(self):
        fhir = self.generator()
        csv = self.generator(
            synthea_config={
                "exporter.csv.export": True,
                "exporter.fhir.export": False,
            }
        )
        both = self.generator(synthea_config={"exporter.csv.export": True})

        self.assertFalse(fhir.csv_enabled)
        self.assertTrue(fhir.fhir_enabled)
        self.assertTrue(csv.csv_enabled)
        self.assertFalse(csv.fhir_enabled)
        self.assertTrue(both.csv_enabled)
        self.assertTrue(both.fhir_enabled)

    def test_invalid_or_unsupported_output_modes_are_rejected(self):
        configs = (
            {"exporter.fhir.export": True, "exporter.fhir.bulk_data": False},
            {"exporter.fhir.export": False, "exporter.fhir.bulk_data": True},
            {"exporter.ccda.export": True},
            {"exporter.json.export": "true"},
        )
        for config in configs:
            with (
                self.subTest(config=config),
                self.assertRaises(ValueError),
            ):
                self.generator(synthea_config=config)

    def test_non_selection_exporter_options_remain_configurable(self):
        generator = self.generator(
            synthea_config={
                "exporter.csv.append_mode": True,
                "exporter.csv.folder_per_run": True,
                "exporter.fhir.use_us_core_ig": False,
                "exporter.ccda.export": False,
            }
        )

        argv = generator.build_argv(Path("java"), Path("synthea.jar"))
        self.assertIn("--exporter.csv.append_mode=true", argv)
        self.assertIn("--exporter.csv.folder_per_run=true", argv)
        self.assertIn("--exporter.fhir.use_us_core_ig=false", argv)
        self.assertIn("--exporter.ccda.export=false", argv)

    def test_folder_per_run_csv_output_is_resolved(self):
        generator = self.generator(
            synthea_config={
                "exporter.csv.export": True,
                "exporter.fhir.export": False,
                "exporter.csv.folder_per_run": True,
            }
        )
        nested = generator.output_path("csv") / "2026_09_20"
        nested.mkdir(parents=True)
        (nested / "patients.csv").write_text("Id\npatient-1\n")

        self.assertEqual(generator.resolved_output_path("csv"), nested)
        self.assertEqual(generator.outputs["csv"], nested)
        self.assertTrue(generator._has_output("csv"))

    def test_base_directory_can_be_overridden(self):
        custom = self.tmp / "custom-output"
        generator = self.generator(
            synthea_config={"exporter.baseDirectory": str(custom)}
        )

        self.assertEqual(generator.generation_dir, custom.resolve())
        argv = generator.build_argv(Path("java"), Path("synthea.jar"))
        self.assertIn(f"--exporter.baseDirectory={custom.resolve()}", argv)

    def test_get_and_show_config_reads_active_jar(self):
        generator = self.generator(
            jar_path=self.properties_jar(),
            synthea_config={
                "generate.thread_pool_size": 4,
                "generate.only_alive_patients": True,
            },
        )

        defaults = generator.get_config()
        effective = generator.get_config(effective=True)

        self.assertEqual(defaults["generate.thread_pool_size"], "-1")
        self.assertIsNone(defaults["exporter.code_map.icd10-cm"])
        self.assertEqual(effective["generate.thread_pool_size"], "4")
        self.assertEqual(effective["exporter.csv.export"], "false")
        self.assertEqual(effective["exporter.fhir.export"], "true")
        self.assertEqual(effective["exporter.fhir.bulk_data"], "true")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            generator.show_config("generate.*")
        rendered = output.getvalue()
        self.assertIn("generate.thread_pool_size = 4 [override]", rendered)
        self.assertIn("generate.only_alive_patients = true [override]", rendered)

    def test_unknown_config_is_checked_against_active_jar(self):
        generator = self.generator(
            jar_path=self.properties_jar(),
            synthea_config={"future.unknown.property": 1},
        )

        with self.assertRaisesRegex(ValueError, "not supported"):
            generator.get_config()

    def test_java_resolution_order(self):
        explicit = self.tmp / "explicit-java"
        explicit.write_text("")
        home = self.tmp / "java-home"
        (home / "bin").mkdir(parents=True)
        (home / "bin" / "java").write_text("")

        with (
            mock.patch.dict(os.environ, {"JAVA_HOME": str(home)}),
            mock.patch(
                "pyhealth.datasets.synthea_generator.shutil.which",
                return_value="/path/java",
            ),
        ):
            self.assertEqual(
                self.generator(java_path=explicit)._resolve_java(),
                explicit.resolve(),
            )
            self.assertEqual(
                self.generator()._resolve_java(),
                (home / "bin" / "java").resolve(),
            )

    def test_missing_java_raises(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch(
                "pyhealth.datasets.synthea_generator.shutil.which",
                return_value=None,
            ),
            self.assertRaisesRegex(RuntimeError, "Java"),
        ):
            self.generator()._resolve_java()

    def test_jar_resolution(self):
        jar = self.tmp / "synthea.jar"
        jar.write_bytes(b"jar")
        self.assertEqual(self.generator(jar_path=jar)._resolve_jar(), jar.resolve())

        with self.assertRaises(FileNotFoundError):
            self.generator(jar_path=self.tmp / "missing.jar")._resolve_jar()

    def test_ensure_generated_reuses_existing_outputs(self):
        generator = self.generator(
            synthea_config={
                "exporter.csv.export": True,
                "exporter.fhir.export": True,
            }
        )
        csv = generator.output_path("csv")
        fhir = generator.output_path("fhir")
        csv.mkdir(parents=True)
        fhir.mkdir(parents=True)
        (csv / "patients.csv").write_text("Id\npatient-1\n")
        (fhir / "Patient.ndjson").write_text('{"resourceType":"Patient"}\n')

        with mock.patch.object(generator, "run") as run:
            generator.ensure_generated()
        run.assert_not_called()

    def test_run_executes_and_validates_requested_outputs(self):
        generator = self.generator(
            synthea_config={
                "exporter.csv.export": True,
                "exporter.fhir.export": True,
            }
        )
        csv = generator.output_path("csv")
        fhir = generator.output_path("fhir")
        csv.mkdir(parents=True)
        fhir.mkdir(parents=True)
        (csv / "patients.csv").write_text("Id\npatient-1\n")
        (fhir / "Patient.ndjson").write_text('{"resourceType":"Patient"}\n')

        with (
            mock.patch.object(generator, "_resolve_java", return_value=Path("java")),
            mock.patch.object(generator, "_resolve_jar", return_value=Path("jar")),
            mock.patch.object(generator, "get_config"),
            mock.patch(
                "pyhealth.datasets.synthea_generator.subprocess.run",
                return_value=mock.Mock(returncode=0),
            ) as subprocess_run,
        ):
            produced = generator.run()

        self.assertEqual(produced, {"csv": csv, "fhir": fhir})
        subprocess_run.assert_called_once_with(
            mock.ANY,
            timeout=None,
            check=False,
        )


class TestSyntheaDatasets(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="synthea_dataset_")
        self.tmp = Path(self._tmp.name)
        self.cache = self.tmp / "cache"

    def tearDown(self):
        self._tmp.cleanup()

    def generator(self, output):
        configs = {
            "csv": {
                "exporter.csv.export": True,
                "exporter.fhir.export": False,
            },
            "fhir": {},
            "both": {
                "exporter.csv.export": True,
                "exporter.fhir.export": True,
            },
        }
        return SyntheaGenerator(
            output_dir=self.tmp / "output",
            synthea_config=configs[output],
            auto_download=False,
        )

    def test_csv_dataset_uses_defaults_or_exact_explicit_tables(self):
        generator = self.generator("csv")
        default = SyntheaDataset(generator, cache_dir=self.cache)
        empty = SyntheaDataset(generator, tables=[], cache_dir=self.cache)
        explicit = SyntheaDataset(
            generator,
            tables=["conditions", "patients", "conditions"],
            cache_dir=self.cache,
        )

        self.assertEqual(default.tables, DEFAULT_TABLES)
        self.assertEqual(empty.tables, DEFAULT_TABLES)
        self.assertEqual(explicit.tables, ["conditions", "patients"])
        self.assertEqual(Path(default.root), generator.output_path("csv"))

    @mock.patch.object(BaseDataset, "load_data", return_value="events")
    def test_csv_dataset_generates_lazily(self, base_load):
        generator = self.generator("csv")
        dataset = SyntheaDataset(generator, tables=["patients"], cache_dir=self.cache)
        root = Path(dataset.root)
        root.mkdir(parents=True)
        (root / "patients.csv").write_text("Id\npatient-1\n")

        with mock.patch.object(generator, "ensure_generated") as ensure:
            self.assertEqual(dataset.load_data(), "events")

        ensure.assert_called_once_with()
        base_load.assert_called_once_with()

    @mock.patch.object(BaseDataset, "load_data", return_value="events")
    def test_csv_dataset_uses_nested_folder_per_run_output(self, base_load):
        generator = self.generator("csv")
        dataset = SyntheaDataset(generator, tables=["patients"], cache_dir=self.cache)
        nested = generator.output_path("csv") / "2026_09_20"
        nested.mkdir(parents=True)
        (nested / "patients.csv").write_text("Id\npatient-1\n")

        with mock.patch.object(generator, "ensure_generated"):
            self.assertEqual(dataset.load_data(), "events")

        self.assertEqual(Path(dataset.root), nested)
        base_load.assert_called_once_with()

    @mock.patch.object(BaseDataset, "load_data", return_value="events")
    def test_csv_default_tables_skip_files_not_emitted_for_empty_types(self, base_load):
        generator = self.generator("csv")
        dataset = SyntheaDataset(generator, cache_dir=self.cache)
        root = Path(dataset.root)
        root.mkdir(parents=True)
        for table in set(DEFAULT_TABLES) - {"allergies", "imaging_studies"}:
            (root / f"{table}.csv").write_text("PATIENT\npatient-1\n")

        with mock.patch.object(generator, "ensure_generated"):
            self.assertEqual(dataset.load_data(), "events")

        self.assertNotIn("allergies", dataset.tables)
        self.assertNotIn("imaging_studies", dataset.tables)
        base_load.assert_called_once_with()

    def test_csv_explicit_missing_table_raises(self):
        generator = self.generator("csv")
        dataset = SyntheaDataset(generator, tables=["allergies"], cache_dir=self.cache)

        with (
            mock.patch.object(generator, "ensure_generated"),
            self.assertRaisesRegex(RuntimeError, "allergies"),
        ):
            dataset.load_data()

    def test_csv_dataset_requires_csv_exporter(self):
        with self.assertRaisesRegex(ValueError, "CSV output"):
            SyntheaDataset(self.generator("fhir"), cache_dir=self.cache)

    def test_preprocess_procedures_supports_legacy_date(self):
        dataset = SyntheaDataset(self.generator("csv"), cache_dir=self.cache)
        frame = nw.from_native(pl.DataFrame({"date": ["2020-01-01T00:00:00Z"]}).lazy())

        result = dataset.preprocess_procedures(frame).collect().to_native()

        self.assertEqual(result["start"].to_list(), ["2020-01-01T00:00:00Z"])

    def test_fhir_dataset_uses_existing_fhir_base_lazily(self):
        generator = self.generator("fhir")
        dataset = SyntheaFHIRDataset(generator, cache_dir=self.cache)

        self.assertIsInstance(dataset, FHIRDataset)
        self.assertEqual(Path(dataset.root), generator.output_path("fhir"))
        self.assertIn("patient", dataset.tables)
        self.assertIn("condition", dataset.tables)

        with (
            mock.patch.object(generator, "ensure_generated") as ensure,
            mock.patch.object(FHIRDataset, "_ensure_prepared_tables") as parent,
        ):
            dataset._ensure_prepared_tables()

        ensure.assert_called_once_with()
        parent.assert_called_once_with()

    def test_fhir_dataset_requires_fhir_exporter(self):
        with self.assertRaisesRegex(ValueError, "Bulk FHIR output"):
            SyntheaFHIRDataset(self.generator("csv"), cache_dir=self.cache)

    def test_synthea_fhir_extracts_codes_from_codeable_concept_lists(self):
        dataset = SyntheaFHIRDataset(self.generator("fhir"), cache_dir=self.cache)
        resources = (
            {
                "resourceType": "CarePlan",
                "id": "cp1",
                "subject": {"reference": "Patient/p1"},
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "734163000",
                            }
                        ]
                    }
                ],
            },
            {
                "resourceType": "ImagingStudy",
                "id": "img1",
                "subject": {"reference": "Patient/p1"},
                "procedureCode": [
                    {
                        "coding": [
                            {
                                "system": "http://loinc.org",
                                "code": "24627-2",
                            }
                        ]
                    }
                ],
            },
        )

        rows = [
            flatten_resource(resource, dataset.resource_specs)
            for resource in resources
        ]

        self.assertEqual(rows[0][0], "care_plan")
        self.assertEqual(
            rows[0][1]["concept_key"],
            "http://snomed.info/sct|734163000",
        )
        self.assertEqual(rows[1][0], "imaging_study")
        self.assertEqual(rows[1][1]["concept_key"], "http://loinc.org|24627-2")

    @unittest.skipUnless(
        os.environ.get("PYHEALTH_SYNTHEA_LIVE"),
        "requires Java and the Synthea jar",
    )
    def test_live_generate_and_load_csv_and_fhir(self):
        generator = SyntheaGenerator(
            output_dir=self.tmp / "live",
            synthea_config={
                "exporter.csv.export": True,
                "exporter.fhir.export": True,
            },
            population=1,
            seed=42,
            timeout=900,
        )
        csv_dataset = SyntheaDataset(generator, cache_dir=self.cache / "csv")
        fhir_dataset = SyntheaFHIRDataset(generator, cache_dir=self.cache / "fhir")

        self.assertEqual(len(csv_dataset.unique_patient_ids), 1)
        self.assertEqual(len(fhir_dataset.unique_patient_ids), 1)


if __name__ == "__main__":
    unittest.main()
