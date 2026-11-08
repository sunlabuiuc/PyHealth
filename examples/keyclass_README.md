# KeyClass phases 4–6

Run from the repository with PyHealth installed:

```sh
python examples/keyclass_synthetic.py
pip install -e '.[keyclass]'
python examples/keyclass_synthetic.py --aggregation snorkel
python -m unittest discover -s tests/core -p test_keyclass.py -v
```

## Architecture and teammate contract

1. `KeywordLabeler.fit(train_texts, descriptions)` extracts English-stopword-filtered
   1–3 grams, embeds phrases/descriptions, assigns each phrase to its nearest
   category by cosine similarity, and keeps top K per category. Ties are stable.
2. `transform(texts)` builds `[N, LFs]` integer votes: -1 abstains, 0..C-1 votes.
3. `aggregate_votes` returns `[N, C]` soft targets and a covered-row mask using
   normalized majority counts or the optional Snorkel categorical LabelModel.
4. `KeyClass` consumes cached `[N, D]` embeddings and soft targets, returning
   `loss`, `logit`, `y_prob`, and `y_true` in the usual PyHealth form. Its default
   MLP has four linear layers with LeakyReLU/dropout between hidden layers.

Teammate rows should supply `patient_id`, `hadm_id`, and `text`. Keep gold
multi-hot `labels` separate for evaluation. Supply an ordered mapping from
stable category IDs to nonempty description strings. `labeler.category_ids_`
is the canonical order for weak labels, predictions, and evaluation labels.
The synthetic example uses three categories; real data can supply all 19.
Never infer the category inventory from the diagnoses present in a split.
Split patients before mining; fit the labeler and aggregator only on training
notes. Preserve row order and IDs when joining embeddings and weak labels.

Use `ClinicalTextEncoder(model_name=<chosen clinical checkpoint>)` in place of
`SyntheticEncoder`. The protocol is `encode(list[str]) -> numpy [N, D]`.
The clinical adapter uses frozen transformer states and masked mean pooling,
without a learned random projection. It batches requests and explicitly
truncates to 512 tokens by default. Record checkpoint and truncation settings.
For real training, cache embeddings and save category order and LF definitions.

Use `tensor` processors for both embeddings and dense probability targets.
The model declares multilabel mode, so it can use PyHealth Trainer for loss
optimization. Do not run classification metrics against soft weak targets;
use a separate gold evaluation dataset with dense multi-hot targets under the
configured label key. Sigmoid predictions are independent; document the
threshold chosen before gold test evaluation.

## Reproduction boundaries

The aggregator's output sums to one and is NOT independent disease prevalence.
This retains the proposal's categorical-supervision/multilabel-output distinction.
All-abstain rows get uniform placeholders and are excluded by `covered` during
initial training. No-match does not mean all diseases are absent. Snorkel requires
at least three LFs; invalid inputs fail explicitly instead of changing methods.
Cross-category disagreement may indicate genuine comorbidity.

This is a synthetic-first implementation of phases 4–6, not a completed numerical
reproduction. Frozen embeddings, mean pooling, truncation, inclusive sklearn
`min_df`, and configurable hidden widths are explicit starting choices. The
paper's strict frequency cutoff, TF-IDF note selection, exact BlueBERT checkpoint,
confidence selection and encoder/self-training schedule remain to be matched.
Rules do not handle negation and correlated phrases can overcount evidence.
To run the planned ablation, change `ngram_range=(1, 1)` versus `(1, 3)` while
holding split, encoder, top_k, seeds and classifier settings fixed.

References: [KeyClass paper](https://arxiv.org/abs/2206.12088v2),
[Snorkel LabelModel](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.model.LabelModel.html).
