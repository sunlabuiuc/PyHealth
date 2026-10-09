from nbkit import code, footer, header, md

FILENAME = "04_choosing_a_model.ipynb"


def cells():
    return [
        *header(
            "04",
            "Choosing a model: from logistic regression to XGBoost",
            [
                "What the main PyHealth models assume about the data, and when to pick each",
                "How to compare several models fairly on the same patients",
                "How to add a gradient-boosted tree baseline (XGBoost) to a PyHealth benchmark",
            ],
            15,
            "Tutorial 03. A GPU runtime adds the sequence models.",
        ),
        md("""
        ## The models at a glance

        All PyHealth models take the dataset in their constructor and read the
        input and output schemas from it, so swapping models is a one-line
        change. The main ones for coded EHR data:

        | Model | Idea | Good first choice when |
        |---|---|---|
        | `LogisticRegression` | Embeds each code, sums them, one linear layer | You want a transparent, hard-to-beat baseline |
        | `MLP` | Same pooled embeddings, then hidden layers | Features interact, but order does not matter |
        | `RNN` | Reads each field as a sequence (GRU or LSTM) | Order within a field carries signal |
        | `Transformer` | Self-attention over each field | Longer histories, long-range interactions |
        | `RETAIN` | Two attention layers over visits | You need visit- and code-level attention weights |
        | `XGBoostModel` | Gradient-boosted trees on code counts or tabular features | Tabular or bag-of-codes data; the standard strong baseline |

        On tabular and bag-of-codes data, gradient-boosted trees are often as
        good as neural models and much faster to train, so a benchmark without
        them can overstate what a neural model adds.

        ## The task: spotting heart failure from the medication list

        We need a question the synthetic data can answer, so we use a
        *phenotyping* task: does this admission include a heart-failure
        diagnosis (ICD-9 428.x), judged from the drugs and procedures recorded
        in it? Phenotyping models like this help find patients with a
        condition when diagnosis codes are missing or unreliable, for example
        when building a study cohort. About 1 in 5 admissions qualifies.

        The task is a few lines (Tutorial 02 explains each part). Drugs are
        kept by name so the results stay readable.
        """),
        code("""
        from collections import defaultdict

        from pyhealth.tasks import BaseTask


        class HeartFailureFromMedications(BaseTask):
            \"\"\"Does an admission include a heart-failure diagnosis (ICD-9 428.x)?\"\"\"

            task_name = "HeartFailureFromMedications"
            input_schema = {"drugs": "sequence", "procedures": "sequence"}
            output_schema = {"heart_failure": "binary"}

            def __call__(self, patient):
                # Fetch each event type once, then group by admission: much
                # faster than one filtered query per admission.
                by_admission = defaultdict(lambda: {"drugs": [], "diagnoses": [], "procedures": []})
                for e in patient.get_events(event_type="prescriptions"):
                    if e.drug:
                        by_admission[e.hadm_id]["drugs"].append(e.drug)
                for e in patient.get_events(event_type="diagnoses_icd"):
                    by_admission[e.hadm_id]["diagnoses"].append(e.icd9_code)
                for e in patient.get_events(event_type="procedures_icd"):
                    by_admission[e.hadm_id]["procedures"].append(e.icd9_code)

                samples = []
                for hadm_id, codes in by_admission.items():
                    if not codes["drugs"] or not codes["diagnoses"]:
                        continue
                    samples.append({
                        "patient_id": patient.patient_id,
                        "visit_id": hadm_id,
                        "drugs": codes["drugs"],
                        "procedures": codes["procedures"] or ["none"],
                        "heart_failure": int(any(c.startswith("428") for c in codes["diagnoses"])),
                    })
                return samples
        """),
        md("""
        ## One split for every model

        A fair comparison uses the same patients for every model, so we build
        the samples once, split by patient (Tutorial 03), and reuse the three
        parts.
        """),
        code("""
        from pyhealth.datasets import MIMIC3Dataset, PatientSplit, get_dataloader

        dataset = MIMIC3Dataset(
            root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III",
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            cache_dir="pyhealth_cache",
        )
        train_ds, val_ds, test_ds = dataset.set_task(
            HeartFailureFromMedications(), split=PatientSplit((0.7, 0.1, 0.2), seed=7)
        )
        print(f"train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}")

        val_loader = get_dataloader(val_ds, batch_size=512)
        test_loader = get_dataloader(test_ds, batch_size=512)
        """),
        md("""
        ## Neural models

        One helper trains any model the same way: same optimizer, same
        maximum epochs, the best epoch picked on validation PR-AUC, and the
        same test set. Only the model changes.

        Sequence models read every drug of every admission in order, which is
        slow on a free CPU runtime (more than 30 minutes each for 35,000
        admissions). On a CPU runtime we therefore train the two pooled
        models; on a GPU runtime (Runtime > Change runtime type > T4 GPU)
        all five neural models run in a few minutes.
        """),
        code("""
        import time

        import numpy as np
        import torch
        from pyhealth.models import MLP, RETAIN, RNN, LogisticRegression, Transformer
        from pyhealth.trainer import Trainer

        METRICS = ["pr_auc", "roc_auc"]
        results = {}


        def fit_and_test(name, model_class, **kwargs):
            torch.manual_seed(0)
            model = model_class(dataset=train_ds, **kwargs)
            trainer = Trainer(model=model, metrics=METRICS, output_path="runs", exp_name=name)
            start = time.time()
            trainer.train(
                train_dataloader=get_dataloader(train_ds, batch_size=256, shuffle=True),
                val_dataloader=val_loader,
                epochs=6,
                monitor="pr_auc",
                patience=2,
            )
            scores = trainer.evaluate(test_loader)
            n_params = sum(p.numel() for p in model.parameters())
            results[name] = {**scores, "seconds": time.time() - start, "parameters": n_params}
            return trainer


        models = [("LogisticRegression", LogisticRegression), ("MLP", MLP)]
        if torch.cuda.is_available():
            models += [("RNN", RNN), ("Transformer", Transformer), ("RETAIN", RETAIN)]
        else:
            print("CPU runtime: training LogisticRegression and MLP only. The sequence models "
                  "(RNN, Transformer, RETAIN) take 30+ minutes each on a free CPU runtime; "
                  "switch to a GPU runtime to include them.")
        for name, cls in models:
            fit_and_test(name, cls)
        """),
        md("""
        ## Adding XGBoost

        Gradient-boosted trees do not train with `Trainer.train`: they are fit
        once on the full training set with `model.fit(...)`. After that they
        work with the same `Trainer.evaluate`, metrics and checkpoints as
        every other model.

        Trees need fixed-width numeric inputs. Code lists vary in length, so
        `bag_of_codes=True` turns each field into per-sample code counts over
        the vocabulary (one column per code). `early_stopping_rounds` uses the
        validation set to choose the number of trees.

        Use unshuffled loaders for `fit`: the order of rows affects row
        subsampling, so a fixed order makes the fit reproducible.
        """),
        code("""
        from pyhealth.models import XGBoostModel

        start = time.time()
        xgb = XGBoostModel(
            train_ds,
            bag_of_codes=True,
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=0,
            early_stopping_rounds=50,
            eval_metric="aucpr",
        )
        xgb.fit(get_dataloader(train_ds, batch_size=1024), val_loader)
        scores = Trainer(model=xgb, metrics=METRICS, enable_logging=False).evaluate(test_loader)
        results["XGBoost"] = {**scores, "seconds": time.time() - start,
                              "parameters": xgb.estimators_[0].best_iteration + 1}
        print(f"{len(xgb.feature_names)} columns, e.g. {xgb.feature_names[:3]}")
        print(f"best number of trees: {xgb.estimators_[0].best_iteration + 1}")
        """),
        md("""
        ## Results
        """),
        code("""
        prevalence = float(np.mean([int(test_ds[i]["heart_failure"]) for i in range(len(test_ds))]))
        print(f"test prevalence (PR-AUC of a random guess): {prevalence:.3f}\\n")
        print(f"{'model':20s} {'PR-AUC':>7s} {'ROC-AUC':>8s} {'seconds':>8s} {'size':>10s}")
        for name, r in sorted(results.items(), key=lambda kv: -kv[1]["pr_auc"]):
            size = f"{r['parameters']} trees" if name == "XGBoost" else f"{r['parameters']:,}"
            print(f"{name:20s} {r['pr_auc']:7.3f} {r['roc_auc']:8.3f} {r['seconds']:8.1f} {size:>10s}")
        """),
        md("""
        ### Reading the table

        - Compare every model with the random-guess line first: PR-AUC is only
          meaningful relative to the prevalence.
        - Look at cost as well as score: training time and model size differ
          by orders of magnitude for similar accuracy.
        - Small gaps can be noise. Before
          declaring a winner, compute bootstrap intervals of the paired
          difference (Tutorial 03, `paired_bootstrap_diff`).
        - Neural model scores also move with the random seed. Report the mean
          and spread over several seeds rather than one run.

        ## Rules of thumb

        1. Start with `LogisticRegression` and `XGBoostModel`. They are fast
           and often strong, and every other model has to beat them.
        2. Move to `RNN`, `Transformer` or `RETAIN` when order or long
           histories plausibly matter, and check the gain with intervals.
        3. Keep the split, the metric used to pick the epoch, and the test set
           identical across models.

        ## Summary

        - Every model reads the schemas, so swapping models is one line.
        - `XGBoostModel` fits with `model.fit(train, val)` and then works with
          the standard `Trainer` evaluation; `bag_of_codes=True` handles code
          lists.
        - Compare on the same patients, report intervals, and include strong
          simple baselines.
        """),
        *footer("05"),
    ]
