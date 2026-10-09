from nbkit import code, footer, header, md

FILENAME = "05_interpreting_predictions.ipynb"


def cells():
    return [
        *header(
            "05",
            "Interpreting predictions",
            [
                "Which inputs drive a model overall (global importance) and for one patient (local explanation)",
                "Exact TreeSHAP for XGBoost, and Integrated Gradients for neural models",
                "How to check whether an explanation is faithful to the model, not just plausible",
            ],
            20,
            "Tutorial 04 (we reuse its heart-failure task).",
        ),
        md("""
        Clinicians reasonably ask *why* a model flags a patient. PyHealth has
        two families of tools:

        - **Exact Shapley values for tree models** (`XGBoostModel.explain`,
          `TreeSHAP`): fast and deterministic, computed by XGBoost itself.
        - **Attribution methods for neural models** in `pyhealth.interpret`:
          Integrated Gradients, DeepLIFT, GIM, attention-based methods (Chefer,
          attention rollout), and model-agnostic SHAP and LIME. They share one
          interface: `attribute(**batch)` returns a score per input position,
          keyed by feature name.

        We use the heart-failure phenotyping task from Tutorial 04: does an
        admission include a heart-failure diagnosis, judged from its drugs and
        procedures?
        """),
        code("""
        from collections import defaultdict

        from pyhealth.datasets import MIMIC3Dataset, PatientSplit, get_dataloader
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


        dataset = MIMIC3Dataset(
            root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III",
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            cache_dir="pyhealth_cache",
        )
        train_ds, val_ds, test_ds = dataset.set_task(
            HeartFailureFromMedications(), split=PatientSplit((0.7, 0.1, 0.2), seed=7)
        )
        val_loader = get_dataloader(val_ds, batch_size=512)
        test_loader = get_dataloader(test_ds, batch_size=512)
        """),
        md("""
        ## 1. Tree models: exact Shapley values

        A Shapley value says how much one input moved a prediction away from
        the average prediction. For tree ensembles they can be computed
        exactly. They are on the **log-odds** scale and add up: the bias plus
        all contributions equals the model's raw score for that sample.

        First, fit the model (as in Tutorial 04):
        """),
        code("""
        from pyhealth.models import XGBoostModel
        from pyhealth.trainer import Trainer

        xgb = XGBoostModel(train_ds, bag_of_codes=True, n_estimators=400, max_depth=4,
                           learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                           random_state=0, early_stopping_rounds=50, eval_metric="aucpr")
        xgb.fit(get_dataloader(train_ds, batch_size=1024), val_loader)
        print(Trainer(model=xgb, metrics=["pr_auc", "roc_auc"], enable_logging=False).evaluate(test_loader))
        """),
        md("""
        ### Global importance

        `mean_abs_shap` averages the size of each column's contribution over a
        dataset: which inputs the model leans on most, in either direction.
        Columns are named `field=code`.
        """),
        code("""
        ranking = xgb.mean_abs_shap(test_loader)
        for name, value in list(ranking.items())[:12]:
            print(f"{value:.3f}  {name}")
        """),
        md("""
        Do these make clinical sense? Diuretics (furosemide), beta-blockers,
        ACE inhibitors and digoxin are standard heart-failure drugs, so we
        would expect them near the top. A ranking that does not make sense is
        worth investigating: it can reveal leakage or a data problem rather
        than medical insight. (The synthetic data only partly preserves real
        prescribing patterns.)

        ### Explaining one admission

        `explain` gives every column's contribution for each sample. Let's take
        the test admission the model is most confident about and list what
        pushed its score up or down.
        """),
        code("""
        import torch

        batch = next(iter(test_loader))
        out = xgb.explain(**batch)
        names = [n for field in xgb.feature_layout for n in out["feature_names"][field["key"]]]
        contrib = torch.cat([out["attributions"][field["key"]] for field in xgb.feature_layout], dim=1)

        i = int(out["logit"][:, 0].argmax())
        prob = torch.sigmoid(out["logit"][i, 0])
        print(f"admission {batch['visit_id'][i]}: predicted heart-failure probability {prob:.2f}, "
              f"true label {int(batch['heart_failure'][i])}")
        order = contrib[i].argsort(descending=True)
        print("\\npushes the score up:")
        for j in order[:6]:
            print(f"  {contrib[i, j]:+.3f}  {names[j]}")
        print("pushes the score down:")
        for j in order[-3:]:
            print(f"  {contrib[i, j]:+.3f}  {names[j]}")
        total = out["bias"][i, 0] + contrib[i].sum() if out["bias"].ndim > 1 else out["bias"][i] + contrib[i].sum()
        print(f"\\nbias + contributions = {float(total):.3f}, model log-odds = {float(out['logit'][i, 0]):.3f}")
        """),
        md("""
        The last line checks the bookkeeping: contributions plus bias equal the
        model output exactly. Note that an *absent* drug can also contribute
        (its absence lowered or raised the score), which is why some names
        above may not appear in this admission's list.

        `pyhealth.interpret.methods.TreeSHAP` returns the same values in the
        shape of each input field, like the neural interpreters below.

        ## 2. Neural models: Integrated Gradients

        For a neural model, Integrated Gradients compares the input with a
        neutral baseline (here, every code replaced by the unknown token) and
        accumulates gradients along the path between them. The result is one
        score per input position: per drug in the list, per procedure.
        """),
        code("""
        from pyhealth.models import MLP

        torch.manual_seed(0)
        mlp = MLP(dataset=train_ds)
        trainer = Trainer(model=mlp, metrics=["pr_auc", "roc_auc"], enable_logging=False)
        trainer.train(
            train_dataloader=get_dataloader(train_ds, batch_size=256, shuffle=True),
            val_dataloader=val_loader,
            epochs=5,
            monitor="pr_auc",
        )
        print(trainer.evaluate(test_loader))
        """),
        code("""
        from pyhealth.interpret.methods import IntegratedGradients

        ig = IntegratedGradients(mlp, steps=32)
        small = next(iter(get_dataloader(test_ds, batch_size=16)))
        attributions = ig.attribute(**small)
        print({k: tuple(v.shape) for k, v in attributions.items()})
        """),
        md("""
        Each attribution tensor has the shape of its input: one score per drug
        position. To read them, map the token ids back to drug names with the
        processor's vocabulary (Tutorial 02):
        """),
        code("""
        vocab = train_ds.input_processors["drugs"].code_vocab
        id_to_drug = {i: d for d, i in vocab.items()}

        with torch.no_grad():
            probs = mlp(**small)["y_prob"][:, 0]
        k = int(probs.argmax())
        print(f"admission {small['visit_id'][k]}: probability {probs[k]:.2f}, true label {int(small['heart_failure'][k])}")
        scores = attributions["drugs"][k].detach()
        tokens = small["drugs"][k]
        ranked = sorted(((float(s), id_to_drug[int(t)]) for s, t in zip(scores, tokens) if int(t) > 1), reverse=True)
        for s, name in ranked[:8]:
            print(f"  {s:+.4f}  {name}")
        """),
        md("""
        ## 3. Is the explanation faithful?

        A plausible-looking explanation is not necessarily the one the model
        uses. A standard check is a **deletion test**: remove the inputs the
        explanation ranks highest and see how far the prediction falls. If the
        explanation is faithful, removing its top drugs should lower the
        predicted risk much more than removing the same number of random
        drugs. (This is the idea behind the *comprehensiveness* metric.)

        We remove a drug by replacing its id with the padding id 0, which the
        model ignores.
        """),
        code("""
        def remove_drugs(batch, scores, k, generator=None):
            \"\"\"Copy of the batch with each sample's k highest-scoring drugs set to padding (always keeps one).\"\"\"
            drugs = batch["drugs"].clone()
            real = drugs > 1                      # skip padding (0) and unknown (1)
            if generator is not None:             # random ranking instead
                scores = torch.rand(drugs.shape, generator=generator)
            ranked = scores.masked_fill(~real, float("-inf")).argsort(dim=1, descending=True)
            for row in range(drugs.shape[0]):
                top = ranked[row, : min(k, int(real[row].sum()) - 1)]  # keep at least one drug
                drugs[row, top] = 0
            return {**batch, "drugs": drugs}


        def risk(batch):
            with torch.no_grad():
                return mlp(**batch)["y_prob"][:, 0]


        eval_batch = next(iter(get_dataloader(test_ds, batch_size=1024)))
        ig_scores = ig.attribute(**eval_batch)["drugs"].detach()
        base = risk(eval_batch)
        flagged = base > 0.5                       # admissions the model calls positive
        gen = torch.Generator().manual_seed(0)
        print(f"{int(flagged.sum())} admissions predicted positive; mean risk {base[flagged].mean():.3f}\\n")
        print("drugs removed   drop with IG ranking   drop with random ranking")
        for k in [1, 3, 5]:
            ig_drop = (base - risk(remove_drugs(eval_batch, ig_scores, k)))[flagged].mean()
            rnd_drop = (base - risk(remove_drugs(eval_batch, ig_scores, k, generator=gen)))[flagged].mean()
            print(f"{k:13d}   {ig_drop:20.3f}   {rnd_drop:24.3f}")
        """),
        md("""
        If the Integrated Gradients column falls clearly faster than the random
        column, the attributions point at inputs the model really relies on.
        `pyhealth.metrics.interpretability` packages this idea as
        comprehensiveness and sufficiency scores for whole datasets.

        """),
        md("""
        ## Choosing a method

        | Situation | Use |
        |---|---|
        | XGBoost or another tree model | `xgb.explain` / `mean_abs_shap` / `TreeSHAP`: exact and fast |
        | Neural model on codes | `IntegratedGradients`, `DeepLift`, `GIM` |
        | Transformer, want attention-based relevance | `CheferRelevance`, `AttentionRollout` |
        | Any model, treat it as a black box | `ShapExplainer`, `LimeExplainer` (slower, sampling-based) |

        Whatever you use, check faithfulness as above, and treat explanations
        as hypotheses to examine with clinicians, not as causal findings: they
        describe the model, and the model reflects the data, biases included.

        ## Summary

        - Tree models give exact Shapley values; they add up to the model's
          log-odds output.
        - Neural interpreters share `attribute(**batch)` and return one score
          per input position; map ids back with the processor vocabulary.
        - Measure faithfulness (comprehensiveness, sufficiency) against a
          random baseline before trusting an explanation.
        """),
        *footer("06"),
    ]
