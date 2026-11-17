# Person C checkpoint handoff — November 18

## Run locally

From this checkout, activate the installed environment with
`source ../.venv/bin/activate`. On another machine install the repository and
pytest in a Python 3.12 or 3.13 environment.

```sh
# Generate all seven populated synthetic CSVs (five invented patients).
python project/person_c/generate_synthetic.py project/person_c/synthetic_eicu
# Fast in-memory task checks; imports still add startup overhead.
python -m pytest tests/core/test_tpc_eicu.py -v --durations=5
# Real CSV loader + preprocessing + LSTM optimizer step.
python -m pytest tests/integration/test_tpc_checkpoint.py -v
# Independently runnable checkpoint, with machine-readable evidence.
python -m project.person_c.checkpoint --output project/person_c/checkpoint_result.json
```

## Delivered

- Five-patient fixtures populate lab, periodic/aperiodic vitals, intake/output,
  respiratory and nursing tables; patient CSV defines admission/static fields.
- Adult/five-hour filtering, four static features and hourly remaining-LOS
  labels; observations stop at hour 336 without capping remaining LOS.
- Optional decay input recomputed after masking, with the authors' formula.
- Independent numerical oracle: `EXPECTED_RESULTS.md`.
- Proposed team contract and reproduction differences: `INTERFACE.md`.
- Tests cover filtering, empty/missing records, static features, labels,
  aggregation/conversions, removal, filling, decay, stay isolation and causality.
- CSV-to-LSTM integration uses actual PyHealth classes. Processors fit only on
  training patients; held-out patients reuse them. Tests require finite loss,
  correct prediction counts, finite nonzero gradients and parameter changes.
- RST API documentation and task index/public import updated.
- Background/implementation slides and 5:40 video outline prepared in this chat.

Evidence: `validation.txt` (test results), `checkpoint_result.json` (training
check). Synthetic losses are only diagnostic; they are NOT clinical results.
No real patient data is used. Temporary integration data and caches are removed.

## Timing

Fast task assertions are measured separately from the CSV integration test.
The latter intentionally exercises PyHealth's Dask server, file conversion,
processors and model. Full command runtime includes imports and cannot be
represented as sub-second. See recorded timings rather than an estimated claim.

## Team integration

Work was based on origin/master e81a557. No newer remote commits were available
at the fetch. Existing calls remain compatible; enable `include_decay=True`
when adding the decay input to A's model. This runs a real LSTM using the public
PyHealth API; A's final training/ablation script is not present in the checkout,
so compatibility with that unseen script cannot be claimed.

Review INTERFACE.md with A/B: units, discharge-zero policy, floor/median bins,
masking hourly cells rather than raw rows, and the feature list are explicit.
The checkpoint provides a tested integration path, not teammate approval.

## Remaining scheduled work

Person C: review the merged A/B code against this contract, update slides with
A's real experiment results, add contributor identities to the final PR draft,
help submit the team PR by November 24, and record the 4–7-minute video by
November 28. Current work follows Option 3 from the work distribution, rather
than the earlier proposal's full TPC model. No upstream PR is submitted here.
Course deadline: November 29, 2026 at 11:59 PM (rubric timezone unspecified).

Video outline: motivation 0:00–0:20; TPC 0:20–1:10; paper results 1:10–1:45;
our scope 1:45–2:30; implementation/tests 2:30–3:30; A's results 3:30–5:00;
limitations/conclusions 5:00–5:40.

AI assistance: code, tests and slide draft were produced with Codex. Review
and understand the changes and accurately disclose assistance in the required
course form. Reference: https://arxiv.org/abs/2007.09483.
