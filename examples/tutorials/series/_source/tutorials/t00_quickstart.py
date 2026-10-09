from nbkit import code, footer, header, md

FILENAME = "00_quickstart.ipynb"


def cells():
    return [
        *header(
            "00",
            "Quickstart: your first clinical prediction model",
            [
                "The five steps every PyHealth project follows: dataset, task, split, model, evaluation",
                "How to predict in-hospital mortality from diagnosis, procedure and drug codes",
                "How to read the results, and why PR-AUC matters for rare outcomes",
            ],
            15,
            "nothing. This is the place to start.",
        ),
        md("""
        ## The PyHealth pipeline in one picture

        Every PyHealth project has the same five steps. Each one is a single
        object, so you can swap a piece (another dataset, task or model)
        without touching the rest:

        | Step | PyHealth object | What it does |
        |---|---|---|
        | 1. Load data | `pyhealth.datasets` (e.g. `MIMIC3Dataset`) | Reads raw tables into patients, each with a timeline of events |
        | 2. Define the task | `pyhealth.tasks` (e.g. `MortalityPredictionMIMIC3`) | Turns each patient into labelled samples (features + label) |
        | 3. Split | `split_by_patient` | Keeps each patient in only one of train / validation / test |
        | 4. Train | `pyhealth.models` + `Trainer` | Fits a model on the training samples |
        | 5. Evaluate | `Trainer.evaluate`, `pyhealth.metrics` | Scores the model on patients it has never seen |

        We will run all five on **synthetic MIMIC-III**: a public, generated
        copy of the MIMIC-III intensive-care database with the same tables and
        columns, but no real patients. Results on synthetic data say nothing
        clinical; the point is to learn the workflow, which is identical on the
        real database.
        """),
        md("""
        ## Step 1. Load the dataset

        `MIMIC3Dataset` reads the raw MIMIC-III tables you ask for and links
        every row to its patient. `root` can be a local folder or a URL; here
        it points at the public synthetic copy.

        - `tables`: which tables to load. Patients and admissions are always
          loaded; we add diagnoses, procedures and prescriptions.
        - `cache_dir`: where PyHealth keeps its processed copy so the next run
          is fast. With real patient data, put this inside storage you control.

        The first run downloads and processes the tables, which takes a couple
        of minutes. PyHealth logs each step as it goes; the long log is normal.
        """),
        code("""
        from pyhealth.datasets import MIMIC3Dataset

        dataset = MIMIC3Dataset(
            root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III",
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            cache_dir="pyhealth_cache",
        )
        dataset.stats()
        """),
        md("""
        A dataset is a collection of **patients**, and each patient is a
        timeline of **events** (an admission, a diagnosis code, a drug order).
        Let's look at one patient:
        """),
        code("""
        patient = dataset.get_patient(dataset.unique_patient_ids[0])
        print("patient:", patient.patient_id)
        for event in patient.get_events()[:5]:
            print(event)
        """),
        md("""
        Tutorial 01 explores datasets in depth. For now, the key idea: the
        dataset holds *all* the data, and it does not yet know what you want to
        predict. That is the task's job.

        ## Step 2. Define the task

        A **task** says what one training example looks like. PyHealth ships
        ready-made tasks for common problems. `MortalityPredictionMIMIC3`
        creates one sample per hospital admission:

        - **Inputs**: the diagnosis codes (`conditions`), procedure codes
          (`procedures`) and drug codes (`drugs`) recorded in that admission.
        - **Label** (`mortality`): 1 if the patient died in hospital during
          their *next* admission, else 0. Patients with a single admission
          have no "next" admission, so they produce no samples.

        `set_task` runs the task over every patient and returns a
        `SampleDataset`, ready for a model.
        """),
        code("""
        from pyhealth.tasks import MortalityPredictionMIMIC3

        task = MortalityPredictionMIMIC3()
        print("inputs:", task.input_schema)
        print("label: ", task.output_schema)

        samples = dataset.set_task(task)
        print(f"{len(samples)} samples")
        """),
        md("""
        The schemas tell PyHealth how to turn each field into numbers: a
        `"sequence"` is a list of codes, mapped to integer ids with a
        vocabulary learned from the data; `"binary"` is a 0/1 label.

        Here is one sample. The codes are already integer ids, and the label is
        a tensor:
        """),
        code("""
        sample = samples[0]
        for key, value in sample.items():
            print(f"{key:12s} {value}")
        """),
        md("""
        Before training, check how common the outcome is. A model that always
        says "survives" would be right most of the time, so accuracy alone is
        misleading for rare outcomes.
        """),
        code("""
        labels = [int(samples[i]["mortality"]) for i in range(len(samples))]
        print(f"{sum(labels)} of {len(labels)} samples are positive "
              f"({sum(labels) / len(labels):.1%})")
        """),
        md("""
        ## Step 3. Split by patient

        One patient can contribute several samples (one per admission). If the
        same patient appears in both training and test data, the model can
        recognise the patient instead of learning anything general, and the
        test score looks better than it really is. `split_by_patient` puts
        every patient in exactly one part: 70% train, 10% validation (to pick
        the best epoch), 20% test (touched only once, at the end).
        """),
        code("""
        from pyhealth.datasets import get_dataloader, split_by_patient

        train_ds, val_ds, test_ds = split_by_patient(samples, [0.7, 0.1, 0.2], seed=42)
        print(f"train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}")

        train_loader = get_dataloader(train_ds, batch_size=64, shuffle=True)
        val_loader = get_dataloader(val_ds, batch_size=64, shuffle=False)
        test_loader = get_dataloader(test_ds, batch_size=64, shuffle=False)
        """),
        md("""
        ## Step 4. Build and train a model

        PyHealth models read the schemas from the dataset, so you do not
        describe the inputs again. `RNN` embeds each code, runs a recurrent
        network over each field, and combines them into one prediction.

        The `Trainer` handles the training loop: batches, optimizer, evaluation
        on the validation set after each epoch, and keeping the best epoch by
        the metric you `monitor`. We monitor **PR-AUC** (area under the
        precision-recall curve), which focuses on how well the rare positive
        cases are found.
        """),
        code("""
        from pyhealth.models import RNN
        from pyhealth.trainer import Trainer

        model = RNN(dataset=samples)
        trainer = Trainer(model=model, metrics=["pr_auc", "roc_auc"])
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=10,
            monitor="pr_auc",
        )
        """),
        md("""
        ## Step 5. Evaluate on the test patients

        The trainer has reloaded the best epoch, so we score it once on the
        held-out test patients.

        How to read the numbers:
        - **ROC-AUC**: the chance that a random positive case is ranked above a
          random negative one. 0.5 is a coin flip, 1.0 is perfect.
        - **PR-AUC**: precision averaged over recall levels. Its no-skill
          baseline is the positive rate printed above, not 0.5, so compare
          against that.
        """),
        code("""
        scores = trainer.evaluate(test_loader)
        prevalence = sum(int(test_ds[i]["mortality"]) for i in range(len(test_ds))) / len(test_ds)
        print(f"test ROC-AUC {scores['roc_auc']:.3f}")
        print(f"test PR-AUC  {scores['pr_auc']:.3f}  (no-skill baseline {prevalence:.3f})")
        """),
        md("""
        On synthetic data the scores are close to chance: the generated
        records have little real signal. On real MIMIC-III the same code learns
        meaningful patterns.

        ## What you did

        ```python
        dataset = MIMIC3Dataset(root=..., tables=[...])           # 1. data
        samples = dataset.set_task(MortalityPredictionMIMIC3())   # 2. task
        train, val, test = split_by_patient(samples, [0.7, 0.1, 0.2])  # 3. split
        trainer = Trainer(model=RNN(dataset=samples))             # 4. model
        trainer.train(...); trainer.evaluate(test_loader)         # 5. evaluate
        ```

        To try a variation, change one line: another task
        (`ReadmissionPredictionMIMIC3`), another model (`Transformer`,
        `RETAIN`), or your own data (Tutorial 01).
        """),
        *footer("01"),
    ]
