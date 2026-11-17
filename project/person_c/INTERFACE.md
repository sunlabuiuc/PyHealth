# Tested interface for A and B

Status: locally implemented and integration-tested against origin/master
`e81a5574cb160798c9040de7b6fa32637913ef74`. This is a concrete proposed contract,
not evidence that teammates have approved it. No newer remote commits were
present when fetched for this revision.

```python
from pyhealth.tasks import TPCLengthOfStayEICU
# Existing A code remains compatible:
task = TPCLengthOfStayEICU(missing_rate=0.1, seed=42)
# Enable tested decay for the full input:
task = TPCLengthOfStayEICU(missing_rate=0.1, seed=42, include_decay=True)
```

One sample per ICU stay: patient_id (unique person), stay_id, dynamic, static,
los, and optional decay. Dynamic is {time: 0..H, values: (H+1,48)}; static is
{time:[0], values:(1,4)}; los is {time:5..H, values:(H-4,1)}; decay mirrors
dynamic's time and dimensions. H=min(floor(total LOS hours),336).

- Use RNN_V2 with `rnn_type="LSTM"` explicitly; its default is GRU.
- Add `decay: [{"values": 8}]` to embedding_dim when include_decay=True.
- Raw targets are HOURS. Convert predictions and targets together to days
  before paper comparisons. Zero labels at exact discharge are preserved.
  The smoke test uses the model's default regression objective; it is not an
  MSLE experiment. A must explicitly select a zero-safe metric/loss convention.
- Preserve existing median/floor hourly binning and YAIB features. The paper
  uses different mean/right-edge binning; this is a documented partial task
  reproduction, not exact original preprocessing.
- Remove observed hourly cells before fitting/imputation. The proposal's raw
  observation deletion differs; don't describe these as identical experiments.
- Fit all learned processors on training patients only and reuse for held-out
  patients. Split by patient_id, never stay_id. The checkpoint uses patient 101
  for training and 103/105 as held out; it is too small for performance claims.
- Static fields are admission values, not discharge values. Missing age is
  excluded; >89 maps to 90. Feature ordering comes from DYNAMIC_FEATURES and
  STATIC_FEATURES rather than a separately maintained list.
- Decay is computed locally per stay, after masking. Default False preserves
  A's prototype. B can retain this implementation or replace it only if the
  expected-value tests still pass or the contract is explicitly revised.

Integration command (from repository root):
`python -m project.person_c.checkpoint --output project/person_c/checkpoint_result.json`
This generates temporary CSVs, reads actual eICUDataset patients, applies the
actual task and processors, runs a true LSTM optimizer step and evaluates the
held-out batch. All temporary data/cache files are cleaned up.
