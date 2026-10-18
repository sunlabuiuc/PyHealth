from nbkit import code, footer, header, md

FILENAME = "08_contributing.ipynb"


def cells():
    return [
        *header(
            "08",
            "Contributing a dataset or task to PyHealth",
            [
                "Whether your idea is a dataset, a task, or both",
                "How a dataset class is built: config, default tables, per-table preprocessing, a default task",
                "How to handle data that is not a table (files on disk)",
                "How to test it with tiny synthetic data, and what a PR needs to pass review",
            ],
            25,
            "Tutorials 01 and 02.",
        ),
        md("""
        ## Dataset or task?

        | You have... | Contribute |
        |---|---|
        | A new data *source* (a database, a public collection) | a dataset in `pyhealth/datasets/` |
        | A new *question* about a source PyHealth already loads | a task in `pyhealth/tasks/` |
        | A new source that comes with its own benchmark question | both, with the task as the dataset's `default_task` |

        A common mistake is to write a new dataset class for a new MIMIC
        question (for example, synthetic data generation from MIMIC-III). That
        is a task: it reuses `MIMIC3Dataset` and adds only the logic of the
        question.

        ## Anatomy of a dataset class

        In Tutorial 01 we loaded CSVs with `BaseDataset` and a YAML config. A
        contributed dataset wraps the same thing in a class, so users only
        pass `root`. Here it is for the small clinic data, with every hook a
        real dataset uses:

        1. A default config shipped with the class.
        2. Default tables that are always loaded (like `patients` and
           `admissions` in MIMIC-III).
        3. `preprocess_<table>`: optional per-table cleaning, applied lazily
           before events are built.
        4. `default_task`: what `dataset.set_task()` runs with no argument.
        """),
        code("""
        from pathlib import Path

        root = Path("my_clinic")
        root.mkdir(exist_ok=True)
        (root / "patients.csv").write_text(
            "patient_id,birth_year,sex\\n" "p1,1956,F\\n" "p2,1971,M\\n" "p3,1990,F\\n"
        )
        (root / "visits.csv").write_text(
            "patient_id,visit_time,diagnosis,systolic_bp\\n"
            "p1,2023-01-05 09:30,i10,152\\n"
            "p1,2023-06-12 14:00,E11.9,148\\n"
            "p1,2024-02-01 10:15,I10,139\\n"
            "p2,2023-03-20 08:45,J45.909,121\\n"
            "p3,2023-11-02 16:20,I10,135\\n"
            "p3,2024-04-18 11:05,n18.3,141\\n"
        )
        config_path = root / "my_clinic.yaml"
        config_path.write_text('''
        version: "1.0"
        tables:
          patients:
            file_path: "patients.csv"
            patient_id: "patient_id"
            timestamp: null
            attributes: ["birth_year", "sex"]
          visits:
            file_path: "visits.csv"
            patient_id: "patient_id"
            timestamp: "visit_time"
            timestamp_format: "%Y-%m-%d %H:%M"
            attributes: ["diagnosis", "systolic_bp"]
        ''')
        print(sorted(p.name for p in root.iterdir()))
        """),
        md("""
        Notice the raw data has inconsistent code case (`i10` vs `I10`). The
        `preprocess_visits` hook fixes that for everyone who uses the dataset.
        Hooks receive a lazy dataframe through
        [narwhals](https://narwhals-dev.github.io/narwhals/), which has the
        same expression syntax as polars; column names arrive lowercased.
        """),
        code("""
        import narwhals as nw
        from pyhealth.datasets import BaseDataset
        from pyhealth.tasks import BaseTask


        class HighBloodPressureNextVisit(BaseTask):
            \"\"\"Is systolic blood pressure >= 140 at the next visit?\"\"\"

            task_name = "HighBloodPressureNextVisit"
            input_schema = {"diagnoses": "sequence", "last_bp": "tensor"}
            output_schema = {"high_bp_next": "binary"}

            def __call__(self, patient):
                visits = patient.get_events(event_type="visits")
                return [
                    {
                        "patient_id": patient.patient_id,
                        "diagnoses": [v.diagnosis for v in visits[: i + 1]],
                        "last_bp": [float(visits[i].systolic_bp)],
                        "high_bp_next": int(float(visits[i + 1].systolic_bp) >= 140),
                    }
                    for i in range(len(visits) - 1)
                ]


        class MyClinicDataset(BaseDataset):
            \"\"\"Visits and demographics from My Clinic.

            Args:
                root: Folder with patients.csv and visits.csv.
                config_path: Optional config; defaults to the one shipped with the class.
                **kwargs: Passed to BaseDataset (cache_dir, dev, num_workers, ...).

            Examples:
                >>> dataset = MyClinicDataset(root="/path/to/my_clinic")  # doctest: +SKIP
                >>> samples = dataset.set_task()  # doctest: +SKIP
            \"\"\"

            def __init__(self, root, config_path=None, **kwargs):
                # In the package this is Path(__file__).parent / "configs" / "my_clinic.yaml".
                config_path = config_path or str(Path(root) / "my_clinic.yaml")
                super().__init__(
                    root=root,
                    tables=["patients", "visits"],   # default tables, always loaded
                    dataset_name="my_clinic",
                    config_path=config_path,
                    **kwargs,
                )

            def preprocess_visits(self, df: nw.LazyFrame) -> nw.LazyFrame:
                \"\"\"Upper-cases diagnosis codes, which the source records inconsistently.\"\"\"
                return df.with_columns(nw.col("diagnosis").str.to_uppercase())

            @property
            def default_task(self):
                return HighBloodPressureNextVisit()


        clinic = MyClinicDataset(root=str(root), cache_dir="pyhealth_cache")
        print([v.diagnosis for v in clinic.get_patient("p1").get_events(event_type="visits")])
        samples = clinic.set_task()
        print(f"{len(samples)} samples; vocabulary {samples.input_processors['diagnoses'].code_vocab}")
        """),
        md("""
        The `i10` from the raw file came out as `I10`, so both spellings share
        one vocabulary entry.

        ## Data that is not a table

        Image, signal and document collections are usually folders of files.
        The trick PyHealth's own loaders use is to **build a metadata table
        once** (one row per file, with its path and labels), save it under
        `root`, and point the config at it. `COVID19CXRDataset` does this in
        `prepare_metadata`, which reads the per-class spreadsheets, builds the
        image paths and labels, checks every file exists, and writes
        `covid19_cxr-metadata-pyhealth.csv`. Its config is then tiny:

        ```yaml
        version: "5.0"
        tables:
          covid19_cxr:
            file_path: "covid19_cxr-metadata-pyhealth.csv"
            patient_id: null      # each row is its own sample
            timestamp: null
            attributes: ["path", "url", "label"]
        ```

        and its task returns `{"image": event.path, "disease": event.label}`
        with an `image` processor that loads the file. The pattern works for
        any file collection.

        ## Testing with tiny synthetic data

        Every contribution needs unit tests under `tests/`. Keep them fast and
        **never commit real patient data**: build a few rows in a temporary
        folder, as above, and check the behaviour you care about. In the
        repository this is a file such as `tests/core/test_my_clinic.py`;
        here we run the same tests inline.
        """),
        code("""
        import tempfile
        import unittest


        class TestMyClinicDataset(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                cls.tmp = tempfile.TemporaryDirectory()
                cls.dataset = MyClinicDataset(root=str(root), cache_dir=cls.tmp.name)

            @classmethod
            def tearDownClass(cls):
                cls.tmp.cleanup()

            def test_codes_are_normalised(self):
                codes = [v.diagnosis for v in self.dataset.get_patient("p3").get_events(event_type="visits")]
                self.assertEqual(codes, ["I10", "N18.3"])

            def test_default_task_labels(self):
                samples = self.dataset.set_task()
                labels = sorted(int(samples[i]["high_bp_next"]) for i in range(len(samples)))
                self.assertEqual(labels, [0, 1, 1])  # p1: 148, 139 -> 1, 0; p3: 141 -> 1

            def test_patient_without_follow_up_has_no_samples(self):
                self.assertEqual(HighBloodPressureNextVisit()(self.dataset.get_patient("p2")), [])


        result = unittest.main(argv=["tutorial"], exit=False, verbosity=2).result
        print("all passed:", result.wasSuccessful())
        """),
        md("""
        ## Opening a pull request

        PyHealth's CI checks every PR against a few rules
        ([CONTRIBUTING.md](https://github.com/sunlabuiuc/PyHealth/blob/master/CONTRIBUTING.md)):

        - [ ] A change under `pyhealth/` comes with a change under `docs/`
              (an API page or an entry in the right index) and under
              `examples/` (a short script that runs your dataset or task).
        - [ ] New or changed public classes and functions have a docstring with
              a `>>>` usage example (see `MyClinicDataset` above).
        - [ ] New code passes `ruff check` (88-character lines).
        - [ ] Tests under `tests/` use synthetic data and pass:
              `python -m unittest discover -t tests -s tests/core`.

        You can run the same rule check locally before pushing:

        ```bash
        python tools/check_pr_rules.py --base origin/master --head HEAD
        ```

        In the PR description, say what the dataset or task is, where the data
        comes from and its license, and how you tested it. A maintainer will
        review it; small, focused PRs are reviewed fastest.

        ## Summary

        - New source: dataset class (config, default tables,
          `preprocess_<table>`, `default_task`). New question: task.
        - Non-tabular data: build a metadata table once, then use the same
          machinery.
        - Test with a few synthetic rows; never commit patient data.
        - PRs need docs, an example, docstring examples, clean lint and
          passing tests.
        """),
        *footer(None),
    ]
