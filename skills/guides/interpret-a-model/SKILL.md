---
name: pyhealth-interpret-a-model
description: Explain what a trained PyHealth model is keying on — feature attributions via IntegratedGradients, DeepLift, Shap, Lime, CheferRelevance, AttentionRollout, GIM, and ensembles — and score how faithful the explanation actually is using comprehensiveness and sufficiency. Use when asked why a model predicted something, which features matter, or to audit a model for spurious signal.
---

# Interpret a model

"Why did the model predict this?" has two honest answers in PyHealth: an **attribution** (a
score per input feature) and a **faithfulness measurement** of that attribution (does removing
the features it called important actually change the prediction?). Produce both. An attribution
on its own is a picture, not evidence — different methods disagree, and a plausible-looking
heatmap can be uncorrelated with what the model is doing.

---

## Step 1 — Check the model is interpretable

Most methods need to reach inside the model, so the model must opt in via a mixin
(`pyhealth/interpret/api.py`).

| Mixin | Models that implement it |
|---|---|
| `Interpretable` (`api.py:5`) | `MLP` (`models/mlp.py:13`), `StageNet` (`models/stagenet.py:243`) |
| `CheferInterpretable` (`api.py:260`) | `Transformer` (`models/transformer.py:324`), `StageAttentionNet` (`models/stagenet_mha.py:301`) |

**Check before promising anything.** If the user's model is an `RNN`, gradient methods that need
`forward_from_embedding` will not work on it as-is. Say so and offer the real options: switch to
a model that implements the interface, use a model-agnostic method (`Lime`, `Shap`), or
implement `Interpretable` on the model (see
[add-a-component](../add-a-component/SKILL.md)).

`Interpretable` splits the model into an embedding stage and a prediction stage so methods can
push gradients through embeddings or perturb them directly. Two requirements it imposes on an
implementing model, both easy to get wrong:

- `forward_from_embedding()` must accept label keys as *optional* kwargs and skip loss
  computation when they are absent.
- Activations must be `nn.Module` instances called as `self.relu(x)` — **not** `F.relu(x)`.
  Functional variants break hook-based attribution.

## Step 2 — Pick a method

All subclass `BaseInterpreter` (`interpret/methods/base_interpreter.py:21`) and share one
contract: construct with the trained model, call `.attribute(**data)`, get back a dict **keyed
by the task's `input_schema` feature names**. So a task with
`input_schema={"conditions": ..., "procedures": ...}` yields
`{"conditions": tensor, "procedures": tensor}`.

| Method | Class | Requires | Notes |
|---|---|---|---|
| Integrated Gradients | `IntegratedGradients` | `Interpretable` | **start here**; axiomatic, needs a baseline |
| DeepLift | `DeepLift` | `Interpretable` | faster than IG, similar output |
| Plain saliency | `BasicGradientSaliencyMaps` | `Interpretable` | cheapest, noisiest |
| Chefer relevance | `CheferRelevance` | `CheferInterpretable` | attention + gradients; for transformers |
| Attention rollout | `AttentionRollout` | `CheferInterpretable` | attention only — weaker evidence |
| GIM | `GIM`, `IntegratedGradientGIM` | `Interpretable` | gradient-informed embedding perturbation |
| SHAP | `ShapExplainer` | model-agnostic | slow; works when the mixin is absent |
| LIME | `LimeExplainer` | model-agnostic | slow; local surrogate |
| Random | `RandomBaseline` | — | **the control** — always run it |
| Ensembles | `AvgEnsemble`, `VarEnsemble`, `CrhEnsemble` | as members | combine methods; `VarEnsemble` exposes disagreement |

All exported from `pyhealth.interpret.methods`.

```python
from pyhealth.interpret.methods import IntegratedGradients

interpreter = IntegratedGradients(model)
attributions = interpreter.attribute(**batch)   # {"conditions": tensor, ...}
```

**Always run `RandomBaseline` alongside.** It is the null hypothesis. If your method's
faithfulness scores do not beat random attribution, the explanation is decoration.

## Step 3 — Measure faithfulness

`pyhealth/metrics/interpretability/` — both metrics subclass `RemovalBasedMetric`
(`interpretability/base.py:21`) and work by deleting features and watching the prediction move.

| Metric | Class | Asks | Good score |
|---|---|---|---|
| Comprehensiveness | `ComprehensivenessMetric` | remove the top-attributed features — does the prediction collapse? | **high** |
| Sufficiency | `SufficiencyMetric` | keep *only* the top features — does the prediction survive? | **low** |

Use `Evaluator` (`interpretability/evaluator.py:19`) or `evaluate_attribution`
(`evaluator.py:407`) to run them, and `threshold_sample_filter`
(`interpretability/utils.py:45`) to restrict the evaluation to confidently-predicted samples —
explaining a sample the model itself is unsure about tells you little.

Report the pair, per method, against the random baseline. That table is the actual deliverable.

Worked examples: `examples/interpretability/` (19 scripts).

---

## What to tell the user

- **Attribution is not causation.** "The model weights this diagnosis code heavily" is a fact
  about the model. It is not a claim about the disease.
- **Methods disagree, and that is information.** Run two or three; where they agree, you have
  something. `VarEnsemble` surfaces the disagreement directly.
- **Show the faithfulness numbers next to the explanation**, always, with the random baseline
  in the same table.
- **High attribution on something clinically absurd is a finding, not a bug to hide.** Leakage
  and spurious correlations show up here first — a mortality model keying on a discharge-coded
  field, or an imaging model keying on a scanner artifact. This is the main reason to run
  interpretability at all.
- Attributions are per-sample. One patient's explanation does not generalize; aggregate across
  a cohort before making a global claim.
