from nbkit import code, footer, header, md

FILENAME = "03_training_and_evaluation.ipynb"


def cells():
    return [
        *header(
            "03",
            "Training and evaluating models properly",
            [
                "How to split so that nothing about the test patients leaks into training",
                "How to control training: epochs, early stopping, learning rate, checkpoints",
                "How to handle a rare outcome with a positive-class weight, and what it costs",
                "How to report discrimination and calibration with patient-level confidence intervals",
            ],
            25,
            "Tutorials 00 and 02.",
        ),
        md("""
        Getting a number out of `trainer.evaluate` is easy. Getting a number
        you can trust, and report, takes a few more steps. This tutorial
        walks through them on the in-hospital mortality task from Tutorial 00,
        a rare outcome (about 3% of samples) where the details matter most.
        """),
        code("""
        from pyhealth.datasets import MIMIC3Dataset
        from pyhealth.tasks import MortalityPredictionMIMIC3

        dataset = MIMIC3Dataset(
            root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III",
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            cache_dir="pyhealth_cache",
        )
        task = MortalityPredictionMIMIC3()
        """),
        md("""
        ## 1. Split by patient, and fit preprocessing on training patients only

        Tutorial 00 already split **by patient**, so no patient appears in two
        parts. There is a second, quieter leak: the processors. When you call
        `set_task(task)` and split afterwards, the code vocabularies are built
        from *all* samples, including the test patients. The model then has an
        embedding slot for a code that only test patients have, which a real
        deployment would never have.

        Passing `split=PatientSplit(...)` to `set_task` splits first and fits
        every processor on the training patients only. It returns one dataset
        per part, training part first. Codes that appear only in validation or
        test map to `<unk>`, exactly as unseen codes would in practice.
        """),
        code("""
        from pyhealth.datasets import PatientSplit

        train_ds, val_ds, test_ds = dataset.set_task(
            task, split=PatientSplit(ratios=(0.7, 0.1, 0.2), seed=42)
        )
        print(f"train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}")
        """),
        md("""
        Compare with fitting on everything first. The vocabulary fitted on all
        samples is larger: the extra entries are codes seen only in the
        validation and test patients.
        """),
        code("""
        all_samples = dataset.set_task(task)

        for field in ["conditions", "procedures", "drugs"]:
            v_all = len(all_samples.input_processors[field].code_vocab)
            v_train = len(train_ds.input_processors[field].code_vocab)
            print(f"{field:11s} vocabulary fitted on all samples {v_all:5d}, on training patients {v_train:5d}")

        unk = train_ds.input_processors["conditions"].code_vocab["<unk>"]
        test_codes = [int(c) for i in range(len(test_ds)) for c in test_ds[i]["conditions"]]
        print(f"{sum(c == unk for c in test_codes) / len(test_codes):.1%} of test diagnosis codes are unseen in training")
        """),
        md("""
        For processors that learn statistics, such as normalisation, the same
        rule applies: statistics from test patients must not shape the
        training inputs. With `split=` you get that for free.

        ## 2. Training with more control

        `Trainer.train` has a few options worth knowing:

        | Option | What it does |
        |---|---|
        | `epochs` | Maximum passes over the training data |
        | `monitor`, `monitor_criterion` | Validation metric that picks the best epoch (`"max"` or `"min"`) |
        | `patience` | Stop early after this many epochs without improvement |
        | `optimizer_params` | e.g. `{"lr": 1e-3}` |
        | `weight_decay`, `max_grad_norm` | Regularisation and gradient clipping |

        The `Trainer` itself takes `output_path` and `exp_name`; the best and
        last checkpoints are saved there, and the best one is reloaded at the
        end of training.
        """),
        code("""
        from pyhealth.datasets import get_dataloader
        from pyhealth.models import RNN
        from pyhealth.trainer import Trainer

        train_loader = get_dataloader(train_ds, batch_size=64, shuffle=True)
        val_loader = get_dataloader(val_ds, batch_size=256)
        test_loader = get_dataloader(test_ds, batch_size=256)

        METRICS = ["pr_auc", "roc_auc"]

        model = RNN(dataset=train_ds)
        trainer = Trainer(model=model, metrics=METRICS, output_path="runs", exp_name="rnn_plain")
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=20,
            monitor="pr_auc",
            patience=5,
            optimizer_params={"lr": 1e-3},
            weight_decay=1e-4,
        )
        """),
        md("""
        ## 3. Rare outcomes: weighting the positive class

        With 3% positives, the loss is dominated by the negatives and a model
        can do well by predicting low risk for everyone. A common remedy is to
        weight each positive example more. `set_pos_weight("balanced",
        train_ds)` sets the weight to negatives / positives in the **training**
        data (never the full dataset, so test labels do not set it).
        """),
        code("""
        weighted = RNN(dataset=train_ds)
        weighted.set_pos_weight("balanced", train_ds)
        print(f"positive weight: {float(weighted.pos_weight[0]):.1f}")

        trainer_w = Trainer(model=weighted, metrics=METRICS, output_path="runs", exp_name="rnn_weighted")
        trainer_w.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=20,
            monitor="pr_auc",
            patience=5,
            optimizer_params={"lr": 1e-3},
            weight_decay=1e-4,
        )
        """),
        md("""
        ## 4. Discrimination *and* calibration

        Ranking patients well (discrimination: ROC-AUC, PR-AUC) is half of
        the story. If the predicted risks will be read as probabilities ("this
        patient has a 20% risk"), they also need to be **calibrated**:

        | Metric | Ideal | Reading it |
        |---|---|---|
        | `brier` | 0 | mean squared error of the probabilities |
        | `oe_ratio` | 1 | observed events / expected events; below 1 means risks are overestimated |
        | `calibration_slope` | 1 | below 1 means predictions are too extreme |
        | `calibration_intercept` | 0 | above 0 means risks are underestimated overall |

        `Trainer.inference` returns the labels and predicted probabilities;
        `binary_metrics_fn` scores them.
        """),
        code("""
        import numpy as np
        from pyhealth.metrics import binary_metrics_fn

        REPORT = ["pr_auc", "roc_auc", "brier", "oe_ratio", "calibration_slope", "calibration_intercept"]

        y_true, prob_plain, _, pids = trainer.inference(test_loader, return_patient_ids=True)
        _, prob_weighted, _ = trainer_w.inference(test_loader)
        y_true, prob_plain, prob_weighted = y_true.ravel(), prob_plain.ravel(), prob_weighted.ravel()

        print(f"test prevalence {y_true.mean():.3f}\\n")
        print(f"{'metric':22s} {'plain':>8s} {'weighted':>9s}")
        plain = binary_metrics_fn(y_true, prob_plain, metrics=REPORT)
        weighted_scores = binary_metrics_fn(y_true, prob_weighted, metrics=REPORT)
        for m in REPORT:
            print(f"{m:22s} {plain[m]:8.3f} {weighted_scores[m]:9.3f}")
        print(f"\\nmean predicted risk: plain {prob_plain.mean():.3f}, weighted {prob_weighted.mean():.3f}")
        """),
        md("""
        Look at the mean predicted risk and the observed / expected ratio:
        weighting pushes every probability up, so the weighted model's risks
        are inflated even when its ranking is similar. That is the price of
        weighting. If you need calibrated risks, either train without the
        weight, or recalibrate on a separate calibration set of patients.

        ## 5. Confidence intervals that respect patients

        A test set of a few hundred samples gives noisy metrics, so report an
        interval. Samples from the same patient are correlated, so the
        bootstrap should resample **whole patients** (`groups=patient_ids`),
        not individual samples. `paired_bootstrap_diff` compares two models on
        the same resampled patients, which is the right way to ask "is model A
        better than model B?".
        """),
        code("""
        from pyhealth.metrics import bootstrap_ci, paired_bootstrap_diff

        groups = np.asarray(pids)
        for name, prob in [("plain", prob_plain), ("weighted", prob_weighted)]:
            for metric in ["roc_auc", "brier"]:
                ci = bootstrap_ci(y_true, prob, metric, groups=groups, n_boot=500, seed=0)
                print(f"{name:9s} {metric:8s} {ci['estimate']:.3f}  95% CI [{ci['lower']:.3f}, {ci['upper']:.3f}]")

        diff = paired_bootstrap_diff(y_true, prob_weighted, prob_plain, "roc_auc", groups=groups, n_boot=500, seed=0)
        print(f"\\nweighted - plain ROC-AUC {diff['estimate']:+.3f}  95% CI [{diff['lower']:+.3f}, {diff['upper']:+.3f}]")
        """),
        md("""
        If the interval of the difference includes 0, the data cannot tell the
        two models apart. With a few hundred test samples and a 3% outcome,
        that is the usual result, and worth knowing before claiming a
        winner.

        ## 6. Saving and reloading a model

        The trainer already saved the best checkpoint under
        `runs/<exp_name>/best.ckpt`. To reuse a model later, build the same
        architecture on a dataset with the same processors, and load it:
        """),
        code("""
        import os

        ckpt = os.path.join(trainer.exp_path, "best.ckpt")
        restored = Trainer(model=RNN(dataset=train_ds), metrics=METRICS, checkpoint_path=ckpt)
        print("restored:", restored.evaluate(test_loader))
        print("original:", trainer.evaluate(test_loader))
        """),
        md("""
        To use the model on new data later, also keep the fitted processors
        (`pyhealth.datasets.save_processors` / `load_processors`), so new
        samples get the same code ids.

        ## Summary

        - Use `set_task(task, split=PatientSplit(...))`: patients stay in one
          part, and preprocessing learns from training patients only.
        - Pick the best epoch on validation with `monitor` and `patience`;
          touch the test set once.
        - A positive-class weight can help rare outcomes but inflates
          probabilities; check calibration.
        - Report discrimination and calibration with patient-level bootstrap
          intervals, and compare models with a paired bootstrap.
        """),
        *footer("04"),
    ]
