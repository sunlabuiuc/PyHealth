import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import narwhals as nw
import polars as pl

from pyhealth.datasets import SyntheaCSVDataset, SyntheaGenerator
from pyhealth.datasets.base_dataset import BaseDataset
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
exporter.ccda.export = false
#exporter.code_map.icd10-cm=code-map.json
""",
            )
        return jar

    def test_generator_produces_csv(self):
        generator = self.generator()

        self.assertNotIsInstance(generator, BaseDataset)
        self.assertEqual(generator.output_path(), generator.generation_dir / "csv")

    def test_generation_configuration_partitions_output(self):
        first = self.generator(seed=1)
        repeated = self.generator(seed=1)
        second = self.generator(seed=2)
        configured = self.generator(
            seed=1,
            synthea_config={"generate.thread_pool_size": 4},
        )

        self.assertEqual(first.generation_dir, repeated.generation_dir)
        self.assertNotEqual(first.generation_dir, second.generation_dir)
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
        invalid_options = (
            ({"clinician_seed": "42"}, TypeError),
            ({"single_person_seed": True}, TypeError),
            ({"reference_date": 20260921}, TypeError),
            ({"end_date": True}, TypeError),
            ({"gender": "X"}, ValueError),
            ({"age_range": (18, 65)}, TypeError),
            ({"overflow_population": 1}, TypeError),
            ({"update_time_period": 0}, ValueError),
        )
        for kwargs, error in invalid_options:
            with self.subTest(kwargs=kwargs), self.assertRaises(error):
                self.generator(**kwargs)

    def test_all_documented_synthea_cli_options_are_forwarded(self):
        local_config = self.tmp / "local.properties"
        local_config.write_text("generate.thread_pool_size = 2\n")
        generator = self.generator(
            population=25,
            seed=42,
            clinician_seed=43,
            single_person_seed=44,
            reference_date="20200102",
            end_date="20251231",
            gender="F",
            age_range="18-65",
            overflow_population=False,
            local_config_path=local_config,
            local_modules_dir=self.tmp / "modules",
            initial_population_snapshot_path=self.tmp / "initial.snapshot",
            updated_population_snapshot_path=self.tmp / "updated.snapshot",
            update_time_period=30,
            fixed_record_path=self.tmp / "fixed.json",
            keep_matching_patients_path=self.tmp / "keep.json",
            state="Massachusetts",
            city="Boston",
            synthea_config={"generate.only_alive_patients": True},
        )

        argv = generator.build_argv(Path("java"), Path("synthea.jar"))

        expected_pairs = (
            ("-c", str(local_config)),
            ("-s", "42"),
            ("-cs", "43"),
            ("-ps", "44"),
            ("-p", "25"),
            ("-r", "20200102"),
            ("-e", "20251231"),
            ("-g", "F"),
            ("-a", "18-65"),
            ("-o", "false"),
            ("-d", str(self.tmp / "modules")),
            ("-i", str(self.tmp / "initial.snapshot")),
            ("-u", str(self.tmp / "updated.snapshot")),
            ("-t", "30"),
            ("-f", str(self.tmp / "fixed.json")),
            ("-k", str(self.tmp / "keep.json")),
        )
        for option, value in expected_pairs:
            with self.subTest(option=option):
                index = argv.index(option)
                self.assertEqual(argv[index + 1], value)
        self.assertEqual(argv[-2:], ["Massachusetts", "Boston"])
        self.assertLess(
            argv.index("-c"), argv.index("--generate.only_alive_patients=true")
        )

    def test_omitted_cli_options_use_synthea_defaults(self):
        argv = self.generator().build_argv(Path("java"), Path("synthea.jar"))

        for option in (
            "-s",
            "-cs",
            "-ps",
            "-p",
            "-r",
            "-e",
            "-g",
            "-a",
            "-o",
            "-c",
            "-d",
            "-i",
            "-u",
            "-t",
            "-f",
            "-k",
        ):
            self.assertNotIn(option, argv)

    def test_csv_argv_only_sets_required_exporter_values(self):
        generator = self.generator(
            population=25,
            seed=42,
            state="Massachusetts",
            city="Boston",
            synthea_config={
                "generate.thread_pool_size": 4,
                "exporter.csv.append_mode": True,
            },
        )

        argv = generator.build_argv(Path("java"), Path("synthea.jar"))

        self.assertIn("--exporter.csv.export=true", argv)
        self.assertIn("--exporter.csv.append_mode=true", argv)
        self.assertNotIn("--exporter.csv.folder_per_run=false", argv)
        self.assertEqual(argv[argv.index("-p") + 1], "25")
        self.assertEqual(argv[argv.index("-s") + 1], "42")
        self.assertEqual(argv[-2:], ["Massachusetts", "Boston"])

    def test_invalid_or_unsupported_output_modes_are_rejected(self):
        configs = (
            {"exporter.csv.export": False},
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
            }
        )

        argv = generator.build_argv(Path("java"), Path("synthea.jar"))
        self.assertIn("--exporter.csv.append_mode=true", argv)
        self.assertIn("--exporter.csv.folder_per_run=true", argv)

    def test_folder_per_run_csv_output_is_resolved(self):
        generator = self.generator(
            synthea_config={
                "exporter.csv.folder_per_run": True,
            }
        )
        nested = generator.output_path() / "2026_09_20"
        nested.mkdir(parents=True)
        (nested / "patients.csv").write_text("Id\npatient-1\n")

        self.assertEqual(generator.resolved_output_path(), nested)
        self.assertTrue(generator._has_output())

    def test_base_directory_can_be_overridden(self):
        custom = self.tmp / "custom-output"
        generator = self.generator(
            synthea_config={"exporter.baseDirectory": str(custom)}
        )

        self.assertEqual(generator.generation_dir, custom.resolve())
        argv = generator.build_argv(Path("java"), Path("synthea.jar"))
        self.assertIn(f"--exporter.baseDirectory={custom.resolve()}", argv)

    def test_config_can_be_discovered_before_construction(self):
        jar = self.properties_jar()

        config = SyntheaGenerator.get_available_config(
            jar_path=jar,
            auto_download=False,
            pattern="generate.*",
        )
        csv_config = SyntheaGenerator.get_available_config(
            jar_path=jar,
            auto_download=False,
            pattern="exporter.csv.*",
        )

        self.assertEqual(config["generate.thread_pool_size"], "-1")
        self.assertNotIn("exporter.csv.export", config)
        self.assertEqual(csv_config["exporter.csv.export"], "false")

        all_config = SyntheaGenerator.get_available_config(
            jar_path=jar,
            auto_download=False,
        )
        self.assertIsNone(all_config["exporter.code_map.icd10-cm"])

    def test_with_config_clones_and_recomputes_derived_state(self):
        generator = self.generator(
            population=10,
            seed=42,
            synthea_config={"generate.only_alive_patients": True},
        )

        configured = generator.with_config({"generate.thread_pool_size": 4})

        self.assertEqual(
            generator.synthea_config, {"generate.only_alive_patients": "true"}
        )
        self.assertNotIn("generate.thread_pool_size", generator.synthea_config)
        self.assertEqual(configured.synthea_config["generate.thread_pool_size"], "4")
        self.assertEqual(configured.population, 10)
        self.assertEqual(configured.seed, 42)
        self.assertNotEqual(configured.generation_dir, generator.generation_dir)
        with self.assertRaises(TypeError):
            generator.synthea_config["generate.thread_pool_size"] = "8"

    def test_local_config_participates_in_fingerprint_and_precedence(self):
        local_config = self.tmp / "local.properties"
        local_config.write_text("generate.only_alive_patients = true\n")
        generator = self.generator(
            jar_path=self.properties_jar(),
            local_config_path=local_config,
            synthea_config={"generate.only_alive_patients": False},
        )

        argv = generator.build_argv(Path("java"), Path("synthea.jar"))

        self.assertNotEqual(generator.generation_dir, self.generator().generation_dir)
        self.assertIn("--generate.only_alive_patients=false", argv)
        self.assertLess(
            argv.index("-c"), argv.index("--generate.only_alive_patients=false")
        )

    def test_unknown_config_is_checked_against_active_jar(self):
        generator = self.generator(
            jar_path=self.properties_jar(),
            synthea_config={"future.unknown.property": 1},
        )

        with self.assertRaisesRegex(ValueError, "not supported"):
            generator._validate_config_keys(generator.jar_path)

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
        generator = self.generator()
        csv = generator.output_path()
        csv.mkdir(parents=True)
        (csv / "patients.csv").write_text("Id\npatient-1\n")

        with mock.patch.object(generator, "run") as run:
            produced = generator.ensure_generated()
        run.assert_not_called()
        self.assertEqual(produced, csv)

    def test_run_executes_and_validates_requested_outputs(self):
        generator = self.generator()
        csv = generator.output_path()
        csv.mkdir(parents=True)
        (csv / "patients.csv").write_text("Id\npatient-1\n")

        with (
            mock.patch.object(generator, "_resolve_java", return_value=Path("java")),
            mock.patch.object(generator, "_resolve_jar", return_value=Path("jar")),
            mock.patch.object(generator, "_validate_config_keys"),
            mock.patch(
                "pyhealth.datasets.synthea_generator.subprocess.run",
                return_value=mock.Mock(returncode=0),
            ) as subprocess_run,
        ):
            produced = generator.run()

        self.assertEqual(produced, csv)
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

    def generator(self):
        return SyntheaGenerator(
            output_dir=self.tmp / "output",
            auto_download=False,
        )

    def test_csv_dataset_uses_defaults_or_exact_explicit_tables(self):
        generator = self.generator()
        default = SyntheaCSVDataset(generator, cache_dir=self.cache)
        empty = SyntheaCSVDataset(generator, tables=[], cache_dir=self.cache)
        explicit = SyntheaCSVDataset(
            generator,
            tables=["conditions", "patients", "conditions"],
            cache_dir=self.cache,
        )

        self.assertEqual(default.tables, DEFAULT_TABLES)
        self.assertEqual(empty.tables, DEFAULT_TABLES)
        self.assertEqual(explicit.tables, ["conditions", "patients"])
        self.assertEqual(Path(default.root), generator.output_path())

    @mock.patch.object(BaseDataset, "load_data", return_value="events")
    def test_csv_dataset_generates_lazily(self, base_load):
        generator = self.generator()
        dataset = SyntheaCSVDataset(
            generator, tables=["patients"], cache_dir=self.cache
        )
        root = Path(dataset.root)
        root.mkdir(parents=True)
        (root / "patients.csv").write_text("Id\npatient-1\n")

        with mock.patch.object(generator, "ensure_generated") as ensure:
            self.assertEqual(dataset.load_data(), "events")

        ensure.assert_called_once_with()
        base_load.assert_called_once_with()

    @mock.patch.object(BaseDataset, "load_data", return_value="events")
    def test_csv_dataset_uses_nested_folder_per_run_output(self, base_load):
        generator = self.generator()
        dataset = SyntheaCSVDataset(
            generator, tables=["patients"], cache_dir=self.cache
        )
        nested = generator.output_path() / "2026_09_20"
        nested.mkdir(parents=True)
        (nested / "patients.csv").write_text("Id\npatient-1\n")

        with mock.patch.object(generator, "ensure_generated"):
            self.assertEqual(dataset.load_data(), "events")

        self.assertEqual(Path(dataset.root), nested)
        base_load.assert_called_once_with()

    @mock.patch.object(BaseDataset, "load_data", return_value="events")
    def test_csv_default_tables_skip_files_not_emitted_for_empty_types(self, base_load):
        generator = self.generator()
        dataset = SyntheaCSVDataset(generator, cache_dir=self.cache)
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
        generator = self.generator()
        dataset = SyntheaCSVDataset(
            generator, tables=["allergies"], cache_dir=self.cache
        )

        with (
            mock.patch.object(generator, "ensure_generated"),
            self.assertRaisesRegex(RuntimeError, "allergies"),
        ):
            dataset.load_data()

    def test_preprocess_procedures_supports_legacy_date(self):
        dataset = SyntheaCSVDataset(self.generator(), cache_dir=self.cache)
        frame = nw.from_native(pl.DataFrame({"date": ["2020-01-01T00:00:00Z"]}).lazy())

        result = dataset.preprocess_procedures(frame).collect().to_native()

        self.assertEqual(result["start"].to_list(), ["2020-01-01T00:00:00Z"])

    @unittest.skipUnless(
        os.environ.get("PYHEALTH_SYNTHEA_LIVE"),
        "requires Java and the Synthea jar",
    )
    def test_live_generate_and_load_csv(self):
        generator = SyntheaGenerator(
            output_dir=self.tmp / "live",
            population=1,
            seed=42,
            timeout=900,
        )
        csv_dataset = SyntheaCSVDataset(generator, cache_dir=self.cache / "csv")

        self.assertEqual(len(csv_dataset.unique_patient_ids), 1)


if __name__ == "__main__":
    unittest.main()
