# GBSG and SUPPORT with SurvivalMDN

These examples connect native datasets and time-to-event tasks to PyHealth's
existing `SurvivalMDN` and standard `Trainer`. They are short integration checks;
their scores do not establish reproduction of a published experiment.

The tasks emit `features`, `duration: [time]` and `event: [indicator]`. After
batching, the two output fields each have shape `[batch, 1]`. Event 1 means an
observed outcome; event 0 means right censoring. Patient identifiers remain
aligned with features and outcomes when training samples are shuffled.

## Local data

GBSG needs a prepared `gbsg.csv`. `prepare_gbsg_csv` can convert the public
DeepSurv HDF5 source when the optional `h5py` package is installed. SUPPORT uses
`Support2Dataset` and a local `support2.csv` whose patient-number header is `sno`.
The complete-case task keeps the eight numeric covariates and encodes the six
categorical covariates as 27 features.

Obtain the authors' released `data/gbsg` and `data/support` directories separately
from [their reference repository](https://github.com/XintianHan/Survival-MDN).
These examples only read local CSV/NPZ inputs; they perform no downloads and do
not redistribute reference code or data.

```bash
python -m examples.survival_pipeline --dataset gbsg --root /data/gbsg \
  --released-dir /data/Survival-MDN/data/gbsg --split 1 --epochs 5 \
  --out /tmp/gbsg-native-pilot
python -m examples.survival_pipeline --dataset support2 --root /data/support2 \
  --released-dir /data/Survival-MDN/data/support --split 1 --epochs 5 \
  --out /tmp/support-native-pilot
```

Run from a source checkout. Choose a new output directory for each pilot;
existing fit directories are rejected and preserved. For a download-free check:

```bash
python -m pytest tests/core/test_survival_pipeline.py -q
```

## Preprocessing and time

The released GBSG protocol keeps raw features and converts months to
`months / cohort_maximum + 0.002`. Its maximum uses the full cohort, including
held-out rows, to reproduce the authors' released convention. This is an
explicit reproduction protocol, not a training-only normalization recipe.

SUPPORT feature means and standard deviations are fitted on the selected
training patients and then applied once to every partition. Times are
`days / 365.25 + 0.001`. No extra standardization or model-side `time_scale` is
applied. The task constructors default to source months/days when those
optional transformations are omitted.

Model predictions require strictly positive time grids in the same units as
the supplied durations. The examples do not call the model with a zero-time
point or change its validation rules.

## Training and restoration

The standard Trainer receives minibatches with no singleton remainder, preserving
all training patients for BatchNorm. Validation uses one complete batch so its
mean loss is the intended patient average. The model supplies its censored
likelihood through `mode=None`; no Trainer subclass is used.

Each pilot records model configuration, time/feature preprocessing, split IDs,
input hashes and runtime/source identity alongside its best checkpoint. The
prediction file includes exact survival probabilities at observed outcome times
for D-calibration, as well as curves used by the time-dependent metrics.

`restore_pilot` reconstructs the recorded model configuration before loading
weights. Supply the same feature preprocessing and duration units; weights alone
do not describe those data transformations. Reload checks compare both plotting
curves and exact observed-time probabilities with their saved values.
