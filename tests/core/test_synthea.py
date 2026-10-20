import os
import shutil
import tempfile
import unittest
import zipfile
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import narwhals as nw
import polars as pl

from pyhealth.datasets import SyntheaCSVDataset
from pyhealth.datasets.base_dataset import BaseDataset
from pyhealth.datasets.synthea_csv import DEFAULT_TABLES
from pyhealth.models import BaseModel, Synthea
from pyhealth.models.generators.synthea import _GenerationSettings


class TestSynthea(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="synthea_")
        self.tmp = Path(self._tmp.name)
        self.output = self.tmp / "output"
        self.cache = self.tmp / "cache"

    def tearDown(self):
        self._tmp.cleanup()

    def generator(self, **kwargs):
        return Synthea(
            output_dir=self.output,
            auto_download=False,
            **kwargs,
        )

    def argv(self, **settings):
        return self.generator().build_argv(Path("java"), Path("synthea.jar"), **settings)

    @staticmethod
    def base_directory(argv):
        prefix = "--exporter.baseDirectory="
        return next(arg[len(prefix) :] for arg in argv if arg.startswith(prefix))

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

    @contextmanager
    def fake_synthea(self, generator):
        """Replaces Java with a stub that writes patients.csv like Synthea."""

        def run(argv, **kwargs):
            csv = Path(self.base_directory(argv)) / "csv"
            csv.mkdir(parents=True, exist_ok=True)
            (csv / "patients.csv").write_text("Id\npatient-1\n")
            return mock.Mock(returncode=0)

        with (
            mock.patch.object(generator, "_resolve_java", return_value=Path("java")),
            mock.patch.object(generator, "_resolve_jar", return_value=Path("jar")),
            mock.patch(
                "pyhealth.models.generators.synthea.subprocess.run",
                side_effect=run,
            ) as subprocess_run,
        ):
            yield subprocess_run

    def test_constructor_only_configures_the_generator(self):
        java = self.tmp / "java"
        generator = self.generator(java_path=java, timeout=60)

        self.assertNotIsInstance(generator, BaseDataset)
        self.assertEqual(generator.output_dir, self.output.resolve())
        self.assertEqual(generator.java_path, java)
        self.assertEqual(generator.timeout, 60)
        self.assertFalse(self.output.exists())
        with self.assertRaises(TypeError):
            Synthea(self.output, population=10)

    def test_is_base_model_without_forward_pass(self):
        generator = self.generator()

        self.assertIsInstance(generator, BaseModel)
        self.assertEqual(
            [name for name, _ in generator.named_parameters()], ["_dummy_param"]
        )
        self.assertIn(f"output_dir={str(self.output.resolve())!r}", repr(generator))
        with self.assertRaisesRegex(NotImplementedError, "generate\\(\\)"):
            generator()

    def test_one_instance_generates_several_populations(self):
        generator = self.generator()

        with self.fake_synthea(generator) as subprocess_run:
            small = generator.generate(population=10, seed=1)
            large = generator.generate(population=50, seed=2)

        self.assertNotEqual(small, large)
        self.assertTrue((small / "patients.csv").is_file())
        self.assertTrue((large / "patients.csv").is_file())
        self.assertEqual(subprocess_run.call_count, 2)
        argvs = [call.args[0] for call in subprocess_run.call_args_list]
        self.assertEqual(argvs[0][argvs[0].index("-p") + 1], "10")
        self.assertEqual(argvs[1][argvs[1].index("-p") + 1], "50")

    def test_generate_reuses_output_unless_overwrite(self):
        generator = self.generator()

        with self.fake_synthea(generator) as subprocess_run:
            first = generator.generate(population=10, seed=1)
            reused = generator.generate(population=10, seed=1)
            self.assertEqual(subprocess_run.call_count, 1)
            overwritten = generator.generate(population=10, seed=1, overwrite=True)

        self.assertEqual(first, reused)
        self.assertEqual(first, overwritten)
        self.assertEqual(subprocess_run.call_count, 2)
        subprocess_run.assert_called_with(mock.ANY, timeout=None, check=False)

    def test_omitted_seed_is_random_so_each_call_is_a_new_population(self):
        generator = self.generator()

        with (
            self.fake_synthea(generator) as subprocess_run,
            self.assertLogs("pyhealth.models.generators.synthea", "INFO") as logs,
        ):
            first = generator.generate(population=10)
            second = generator.generate(population=10)

        self.assertNotEqual(first, second)
        self.assertEqual(subprocess_run.call_count, 2)
        argvs = [call.args[0] for call in subprocess_run.call_args_list]
        seeds = [argv[argv.index("-s") + 1] for argv in argvs]
        self.assertNotEqual(seeds[0], seeds[1])
        self.assertIn(f"using seed={seeds[0]}", "\n".join(logs.output))

    def test_generate_raises_when_synthea_fails_or_writes_nothing(self):
        generator = self.generator()
        patches = (
            mock.patch.object(generator, "_resolve_java", return_value=Path("java")),
            mock.patch.object(generator, "_resolve_jar", return_value=Path("jar")),
        )
        with patches[0], patches[1]:
            with (
                mock.patch(
                    "pyhealth.models.generators.synthea.subprocess.run",
                    return_value=mock.Mock(returncode=1),
                ),
                self.assertRaisesRegex(RuntimeError, "status 1"),
            ):
                generator.generate(seed=1)
            with (
                mock.patch(
                    "pyhealth.models.generators.synthea.subprocess.run",
                    return_value=mock.Mock(returncode=0),
                ),
                self.assertRaisesRegex(RuntimeError, "did not generate"),
            ):
                generator.generate(seed=1)

    def test_settings_partition_output(self):
        first = self.base_directory(self.argv(seed=1))
        repeated = self.base_directory(self.argv(seed=1))
        second = self.base_directory(self.argv(seed=2))
        configured = self.base_directory(
            self.argv(seed=1, synthea_config={"generate.thread_pool_size": 4})
        )

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, configured)
        self.assertEqual(Path(first).parent, self.output.resolve())

    def test_setting_validation(self):
        for population in (0, -1):
            with self.subTest(population=population), self.assertRaises(ValueError):
                self.argv(population=population)
        for population in (True, "10"):
            with self.subTest(population=population), self.assertRaises(TypeError):
                self.argv(population=population)
        with self.assertRaises(TypeError):
            self.argv(seed="42")
        with self.assertRaisesRegex(ValueError, "city requires state"):
            self.argv(city="Boston")
        with self.assertRaises(TypeError):
            self.argv(unknown_setting=1)
        invalid_options = (
            ({"clinician_seed": "42"}, TypeError),
            ({"single_person_seed": True}, TypeError),
            ({"reference_date": 20260921}, TypeError),
            ({"end_date": True}, TypeError),
            ({"gender": "X"}, ValueError),
            ({"age_range": (18, 65)}, TypeError),
            ({"overflow_population": 1}, TypeError),
            ({"update_time_period": 0}, ValueError),
            ({"synthea_config": [("a", 1)]}, TypeError),
        )
        for kwargs, error in invalid_options:
            with self.subTest(kwargs=kwargs), self.assertRaises(error):
                self.argv(**kwargs)

    def test_generate_validates_before_running(self):
        generator = self.generator()

        with (
            self.fake_synthea(generator) as subprocess_run,
            self.assertRaisesRegex(ValueError, "city requires state"),
        ):
            generator.generate(city="Boston")
        subprocess_run.assert_not_called()

    def test_all_documented_synthea_cli_options_are_forwarded(self):
        local_config = self.tmp / "local.properties"
        local_config.write_text("generate.thread_pool_size = 2\n")
        argv = self.argv(
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
        argv = self.argv()

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
        argv = self.argv(
            population=25,
            seed=42,
            state="Massachusetts",
            city="Boston",
            synthea_config={
                "generate.thread_pool_size": 4,
                "exporter.csv.append_mode": True,
            },
        )

        self.assertIn("--exporter.csv.export=true", argv)
        self.assertIn("--exporter.csv.append_mode=true", argv)
        self.assertNotIn("--exporter.csv.folder_per_run=false", argv)
        self.assertEqual(argv[argv.index("-p") + 1], "25")
        self.assertEqual(argv[argv.index("-s") + 1], "42")
        self.assertEqual(argv[-2:], ["Massachusetts", "Boston"])

    def test_managed_output_settings_are_rejected(self):
        configs = (
            {"exporter.csv.export": False},
            {"exporter.ccda.export": True},
            {"exporter.json.export": "true"},
            {"exporter.baseDirectory": str(self.tmp / "custom-output")},
        )
        for config in configs:
            with (
                self.subTest(config=config),
                self.assertRaises(ValueError),
            ):
                self.argv(synthea_config=config)

    def test_non_selection_exporter_options_remain_configurable(self):
        argv = self.argv(
            synthea_config={
                "exporter.csv.append_mode": True,
                "exporter.csv.folder_per_run": True,
            }
        )

        self.assertIn("--exporter.csv.append_mode=true", argv)
        self.assertIn("--exporter.csv.folder_per_run=true", argv)

    def test_folder_per_run_csv_output_is_resolved(self):
        generation_dir = self.output / "population"
        self.assertIsNone(Synthea._find_csv_dir(generation_dir))

        nested = generation_dir / "csv" / "2026_09_20"
        nested.mkdir(parents=True)
        (nested / "patients.csv").write_text("Id\npatient-1\n")

        self.assertEqual(Synthea._find_csv_dir(generation_dir), nested)

    def test_config_can_be_discovered_before_construction(self):
        jar = self.properties_jar()

        config = Synthea.get_available_config(
            jar_path=jar,
            auto_download=False,
            pattern="generate.*",
        )
        csv_config = Synthea.get_available_config(
            jar_path=jar,
            auto_download=False,
            pattern="exporter.csv.*",
        )

        self.assertEqual(config["generate.thread_pool_size"], "-1")
        self.assertNotIn("exporter.csv.export", config)
        self.assertEqual(csv_config["exporter.csv.export"], "false")

        all_config = Synthea.get_available_config(
            jar_path=jar,
            auto_download=False,
        )
        self.assertIsNone(all_config["exporter.code_map.icd10-cm"])

    def test_local_config_participates_in_fingerprint_and_precedence(self):
        local_config = self.tmp / "local.properties"
        local_config.write_text("generate.only_alive_patients = true\n")
        argv = self.argv(
            local_config_path=local_config,
            synthea_config={"generate.only_alive_patients": False},
        )

        self.assertNotEqual(self.base_directory(argv), self.base_directory(self.argv()))
        self.assertIn("--generate.only_alive_patients=false", argv)
        self.assertLess(
            argv.index("-c"), argv.index("--generate.only_alive_patients=false")
        )
        with self.assertRaises(FileNotFoundError):
            self.argv(local_config_path=self.tmp / "missing.properties")

    def test_unknown_config_is_checked_against_active_jar(self):
        settings = _GenerationSettings(synthea_config={"future.unknown.property": 1})

        with self.assertRaisesRegex(ValueError, "not supported"):
            Synthea._validate_config_keys(self.properties_jar(), settings)

    def test_java_resolution_order(self):
        explicit = self.tmp / "explicit-java"
        explicit.write_text("")
        home = self.tmp / "java-home"
        (home / "bin").mkdir(parents=True)
        (home / "bin" / "java").write_text("")

        with (
            mock.patch.dict(os.environ, {"JAVA_HOME": str(home)}),
            mock.patch(
                "pyhealth.models.generators.synthea.shutil.which",
                return_value="/path/java",
            ),
            mock.patch.object(Synthea, "_java_major_version", return_value=17),
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
                "pyhealth.models.generators.synthea.shutil.which",
                return_value=None,
            ),
            self.assertRaisesRegex(RuntimeError, "Java"),
        ):
            self.generator()._resolve_java()

    def test_java_version_parsing(self):
        outputs = (
            ('openjdk version "17.0.20.1" 2026-08-18\n', 17),
            ('openjdk version "21" 2023-09-19\n', 21),
            ('openjdk version "25-ea" 2025-09-16\n', 25),
            ('java version "1.8.0_392"\n', 8),
            ("Error: could not find libjava.so\n", None),
        )
        for stderr, expected in outputs:
            with (
                self.subTest(stderr=stderr),
                mock.patch(
                    "pyhealth.models.generators.synthea.subprocess.run",
                    return_value=mock.Mock(stderr=stderr, stdout=""),
                ),
            ):
                self.assertEqual(Synthea._java_major_version(Path("java")), expected)

    def test_old_java_raises_with_its_version(self):
        old_java = self.tmp / "java"
        old_java.write_text("")

        with (
            mock.patch.object(Synthea, "_java_major_version", return_value=11),
            self.assertRaisesRegex(RuntimeError, "is Java 11"),
        ):
            self.generator(java_path=old_java)._resolve_java()

    def test_unknown_java_version_is_tolerated(self):
        java = self.tmp / "java"
        java.write_text("")

        with mock.patch.object(Synthea, "_java_major_version", return_value=None):
            self.assertEqual(
                self.generator(java_path=java)._resolve_java(), java.resolve()
            )

    def test_unrunnable_java_raises(self):
        with (
            mock.patch(
                "pyhealth.models.generators.synthea.subprocess.run",
                side_effect=PermissionError("not executable"),
            ),
            self.assertRaisesRegex(RuntimeError, "cannot run Java"),
        ):
            Synthea._java_major_version(Path("java"))

    @unittest.skipUnless(shutil.which("java"), "requires Java on PATH")
    def test_real_java_version_is_detected(self):
        version = Synthea._java_major_version(Path(shutil.which("java")))

        self.assertIsInstance(version, int)

    def test_jar_resolution(self):
        jar = self.tmp / "synthea.jar"
        jar.write_bytes(b"jar")
        self.assertEqual(self.generator(jar_path=jar)._resolve_jar(), jar.resolve())

        with self.assertRaises(FileNotFoundError):
            self.generator(jar_path=self.tmp / "missing.jar")._resolve_jar()


class TestSyntheaDatasets(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="synthea_dataset_")
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "csv"
        self.root.mkdir()
        self.cache = self.tmp / "cache"

    def tearDown(self):
        self._tmp.cleanup()

    def test_csv_dataset_uses_defaults_or_exact_explicit_tables(self):
        default = SyntheaCSVDataset(self.root, cache_dir=self.cache)
        empty = SyntheaCSVDataset(self.root, tables=[], cache_dir=self.cache)
        explicit = SyntheaCSVDataset(
            self.root,
            tables=["conditions", "patients", "conditions"],
            cache_dir=self.cache,
        )

        self.assertEqual(default.tables, DEFAULT_TABLES)
        self.assertEqual(empty.tables, DEFAULT_TABLES)
        self.assertEqual(explicit.tables, ["conditions", "patients"])
        self.assertEqual(Path(default.root), self.root)

    @mock.patch.object(BaseDataset, "load_data", return_value="events")
    def test_csv_dataset_loads_existing_tables(self, base_load):
        (self.root / "patients.csv").write_text("Id\npatient-1\n")
        dataset = SyntheaCSVDataset(
            self.root, tables=["patients"], cache_dir=self.cache
        )

        self.assertEqual(dataset.load_data(), "events")
        base_load.assert_called_once_with()

    @mock.patch.object(BaseDataset, "load_data", return_value="events")
    def test_csv_default_tables_skip_files_not_emitted_for_empty_types(self, base_load):
        for table in set(DEFAULT_TABLES) - {"allergies", "imaging_studies"}:
            (self.root / f"{table}.csv").write_text("PATIENT\npatient-1\n")
        dataset = SyntheaCSVDataset(self.root, cache_dir=self.cache)

        self.assertEqual(dataset.load_data(), "events")

        self.assertNotIn("allergies", dataset.tables)
        self.assertNotIn("imaging_studies", dataset.tables)
        base_load.assert_called_once_with()

    def test_csv_explicit_missing_table_raises(self):
        dataset = SyntheaCSVDataset(
            self.root, tables=["allergies"], cache_dir=self.cache
        )

        with self.assertRaisesRegex(RuntimeError, "allergies"):
            dataset.load_data()

    def test_preprocess_procedures_supports_legacy_date(self):
        dataset = SyntheaCSVDataset(self.root, cache_dir=self.cache)
        frame = nw.from_native(pl.DataFrame({"date": ["2020-01-01T00:00:00Z"]}).lazy())

        result = dataset.preprocess_procedures(frame).collect().to_native()

        self.assertEqual(result["start"].to_list(), ["2020-01-01T00:00:00Z"])

    @unittest.skipUnless(
        os.environ.get("PYHEALTH_SYNTHEA_LIVE"),
        "requires Java and the Synthea jar",
    )
    def test_live_generate_and_load_csv(self):
        synthea = Synthea(output_dir=self.tmp / "live", timeout=900)
        csv_dataset = SyntheaCSVDataset(
            synthea.generate(population=1, seed=42), cache_dir=self.cache / "csv"
        )

        self.assertEqual(len(csv_dataset.unique_patient_ids), 1)


if __name__ == "__main__":
    unittest.main()
