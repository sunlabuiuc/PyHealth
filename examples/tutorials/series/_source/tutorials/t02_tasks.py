from nbkit import code, footer, header, md

FILENAME = "02_tasks_and_processors.ipynb"


def cells():
    return [
        *header(
            "02",
            "Tasks and processors: from patients to model-ready samples",
            [
                "What a task is: a function from one patient to a list of samples",
                "How input and output schemas pick a processor for each field",
                "What each common processor produces (sequence, nested sequence, multi-hot, tensor, labels)",
                "How to write your own task, filter patients early, and use it with any model",
            ],
            25,
            "Tutorial 01 (Datasets).",
        ),
        md("""
        ## Where tasks fit

        ![Tasks build on datasets](https://drive.google.com/uc?export=view&id=1hHJcavXqisH9JEMqEVtE4TqEg_E489l5)

        A dataset holds every patient's events. A **task** answers one
        question about them: for each patient it returns zero or more
        **samples**, each a dictionary of input features and a label.

        Every task is a small class with three parts:

        ```python
        class MyTask(BaseTask):
            task_name = "my_task"                          # used in cache names
            input_schema = {"conditions": "sequence"}      # feature -> processor
            output_schema = {"label": "binary"}            # label -> processor

            def __call__(self, patient) -> list[dict]:     # one patient -> samples
                ...
        ```

        The **schemas** name a *processor* for each field. Processors turn raw
        values (lists of codes, numbers, text) into tensors, learning what they
        need (such as a code vocabulary) from the data. Think of them as the
        tokenizer of a language model, applied to each field.
        """),
        md("""
        ## Loading the data

        Same dataset as Tutorial 01, without the notes table.
        """),
        code("""
        from pyhealth.datasets import MIMIC3Dataset

        dataset = MIMIC3Dataset(
            root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III",
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            cache_dir="pyhealth_cache",
        )
        """),
        md("""
        ## Processors, one at a time

        Before writing a task, let's see what the common processors do. We
        build a toy dataset of three hand-written samples with
        `create_sample_dataset`, which applies the processors exactly as
        `set_task` does, then print what came out.
        """),
        code("""
        from pyhealth.datasets import create_sample_dataset

        toy = [
            {"patient_id": "a", "codes": ["I10", "E11"], "visits": [["I10"], ["E11", "N18"]],
             "flags": ["female", "smoker"], "labs": [1.2, 140.0], "label": 1},
            {"patient_id": "b", "codes": ["J45"], "visits": [["J45"]],
             "flags": ["male"], "labs": [0.9, 138.0], "label": 0},
            {"patient_id": "c", "codes": ["I10", "J45", "N18"], "visits": [["I10", "J45"], ["N18"], ["I10"]],
             "flags": ["female"], "labs": [1.5, 135.0], "label": 1},
        ]
        toy_ds = create_sample_dataset(
            samples=toy,
            input_schema={"codes": "sequence", "visits": "nested_sequence",
                          "flags": "multi_hot", "labs": "tensor"},
            output_schema={"label": "binary"},
        )
        for key, value in toy_ds[2].items():
            print(f"{key:8s} {value}")
        """),
        md("""
        Reading the output for patient `c`:

        | Field | Processor | Raw value | Becomes |
        |---|---|---|---|
        | `codes` | `sequence` | a list of codes | one integer id per code (ids 0 and 1 are reserved for padding and unknown codes) |
        | `visits` | `nested_sequence` | a list of visits, each a list of codes | a 2-D tensor, one row per visit, padded to the longest visit |
        | `flags` | `multi_hot` | a set of categories | a 0/1 vector over all categories seen |
        | `labs` | `tensor` | numbers | a float tensor, unchanged |
        | `label` | `binary` | 0/1 | a float tensor of shape (1,) |

        The vocabularies the processors learned are kept with the dataset, so
        you can always map ids back to codes:
        """),
        code("""
        codes_vocab = toy_ds.input_processors["codes"].code_vocab
        print("codes vocabulary:", codes_vocab)
        print("flags categories:", toy_ds.input_processors["flags"].label_vocab)

        id_to_code = {i: c for c, i in codes_vocab.items()}
        print("patient c's codes:", [id_to_code[int(i)] for i in toy_ds[2]["codes"]])
        """),
        md("""
        Other processors you will meet:

        | Processor | Use it for |
        |---|---|
        | `multiclass`, `multilabel`, `regression` | labels with several classes, several labels at once, or a number |
        | `timeseries` | irregular measurements resampled to a fixed grid |
        | `text` | free text, for language models |
        | `nested_multihot` | per-visit code sets as 0/1 rows (compact for large vocabularies) |
        | `image`, `audio`, `signal` | file paths to images, audio, physiological signals |
        | `raw` | values a model reads as Python objects (kept unchanged) |

        The full list is in the
        [processors docs](https://pyhealth.readthedocs.io/en/latest/api/processors.html).

        ## Using a built-in task

        PyHealth ships tasks for common questions on each dataset: mortality,
        readmission, length of stay, drug recommendation, medical coding and
        more. Built-in tasks can take options; readmission, for example, lets
        you choose the time window and whether to drop admissions of minors.

        Options matter: by default this task skips patients under 18. The
        synthetic data shifts every birth date to just before the first
        admission, so every patient looks like a newborn and the default would
        return no samples at all. On real MIMIC-III keep the default; here we
        turn it off. When a task returns nothing, check its options and the
        values it reads before suspecting the data.
        """),
        code("""
        from datetime import timedelta
        from pyhealth.tasks import ReadmissionPredictionMIMIC3

        readmission = ReadmissionPredictionMIMIC3(window=timedelta(days=30), exclude_minors=False)
        print(readmission.input_schema, "->", readmission.output_schema)

        readmit_samples = dataset.set_task(readmission)
        positives = sum(int(readmit_samples[i]["readmission"]) for i in range(len(readmit_samples)))
        print(f"{len(readmit_samples)} samples, {positives} readmitted within 30 days")
        """),
        md("""
        ## Writing your own task

        Suppose we want to predict a **long hospital stay** (more than 7 days)
        at admission time, from:

        - the diagnosis codes of the current admission (`sequence`),
        - the diagnosis codes of every admission so far, visit by visit
          (`nested_sequence`), so a model can see the history,
        - the admission type and the patient's sex (`multi_hot`),
        - the number of earlier admissions (`tensor`).

        The `__call__` method is plain Python over the patient's events, using
        the same `get_events` calls as Tutorial 01.

        `pre_filter` is optional. It runs once on the whole event table before
        any patient is processed, so it is a cheap place to drop patients the
        task can never use. Here we keep patients with at least one diagnosis.
        """),
        code("""
        from datetime import datetime

        import polars as pl
        from pyhealth.tasks import BaseTask


        class LongStayMIMIC3(BaseTask):
            \"\"\"Predicts whether an admission lasts more than 7 days.\"\"\"

            task_name = "LongStayMIMIC3"
            input_schema = {
                "conditions": "sequence",
                "history": "nested_sequence",
                "demographics": "multi_hot",
                "prior_admissions": "tensor",
            }
            output_schema = {"long_stay": "binary"}

            def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
                with_diagnoses = (
                    df.filter(pl.col("event_type") == "diagnoses_icd").select("patient_id").unique()
                )
                return df.join(with_diagnoses, on="patient_id", how="semi")

            def __call__(self, patient):
                sex = patient.get_events(event_type="patients")[0].gender
                history = []
                samples = []
                for n, adm in enumerate(patient.get_events(event_type="admissions")):
                    codes = [e.icd9_code for e in patient.get_events(
                        event_type="diagnoses_icd", filters=[("hadm_id", "==", adm.hadm_id)])]
                    if not codes or adm.dischtime is None:
                        continue
                    history.append(codes)
                    discharge = datetime.strptime(adm.dischtime, "%Y-%m-%d %H:%M:%S")
                    stay_days = (discharge - adm.timestamp).total_seconds() / 86400
                    samples.append({
                        "patient_id": patient.patient_id,
                        "visit_id": adm.hadm_id,
                        "conditions": codes,
                        "history": list(history),
                        "demographics": [f"sex={sex}", f"type={adm.admission_type}"],
                        "prior_admissions": [float(n)],
                        "long_stay": int(stay_days > 7),
                    })
                return samples
        """),
        md("""
        Two habits worth keeping in your own tasks:

        - **Only use information available at prediction time.** The inputs
          above come from the current and earlier admissions; the label comes
          from the discharge time, which is in the future when the prediction
          is made. A feature such as "number of drugs given during the stay"
          would leak the answer.
        - **Test on one patient first.** A task is just a function, so call it
          directly before running it on everyone:
        """),
        code("""
        task = LongStayMIMIC3()
        for pid in dataset.unique_patient_ids[:200]:
            out = task(dataset.get_patient(pid))
            if len(out) > 1:
                for s in out:
                    print({k: s[k] for k in ("visit_id", "conditions", "demographics", "prior_admissions", "long_stay")})
                break
        """),
        code("""
        long_stay = dataset.set_task(task)
        print(f"{len(long_stay)} samples")
        sample = long_stay[0]
        for key in task.input_schema | task.output_schema:
            print(f"{key:17s} {tuple(sample[key].shape)}  {sample[key]}")
        """),
        md("""
        ## The task works with any model

        Because the schemas describe the inputs, models need no extra
        configuration. Here `MultimodalRNN` routes the sequence fields through
        recurrent layers and the multi-hot and numeric fields through linear
        layers. (Tutorial 03 covers training properly; this is a smoke test.)
        """),
        code("""
        from pyhealth.datasets import get_dataloader, split_by_patient
        from pyhealth.models import MultimodalRNN
        from pyhealth.trainer import Trainer

        train_ds, val_ds, test_ds = split_by_patient(long_stay, [0.7, 0.1, 0.2], seed=0)
        trainer = Trainer(model=MultimodalRNN(dataset=long_stay), metrics=["roc_auc", "pr_auc"])
        trainer.train(
            train_dataloader=get_dataloader(train_ds, batch_size=64, shuffle=True),
            val_dataloader=get_dataloader(val_ds, batch_size=64),
            epochs=3,
            monitor="roc_auc",
        )
        print(trainer.evaluate(get_dataloader(test_ds, batch_size=64)))
        """),
        md("""
        ## Tasks beyond patient timelines

        A "patient" is just whatever a dataset groups events by. For image or
        document collections, each row can be its own patient, and the task
        returns one sample per row. PyHealth's chest X-ray task, for example,
        is only a few lines:

        ```python
        class COVID19CXRClassification(BaseTask):
            task_name = "COVID19CXRClassification"
            input_schema = {"image": "image"}        # a file path -> image tensor
            output_schema = {"disease": "multiclass"}

            def __call__(self, patient):
                event = patient.get_events(event_type="covid19_cxr")[0]
                return [{"image": event.path, "disease": event.label}]
        ```

        Tutorial 07 uses the same pattern for clinical text.

        ## Summary

        - A task maps one patient to a list of sample dictionaries.
        - `input_schema` / `output_schema` choose a processor per field; the
          processors learn vocabularies and produce tensors.
        - `pre_filter` drops unusable patients early; call the task on one
          patient to debug it.
        - Keep features to what is known at prediction time.
        """),
        *footer("03"),
    ]
