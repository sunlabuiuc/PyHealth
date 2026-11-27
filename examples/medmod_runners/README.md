# MedMod runner scripts

These scripts produced the real-data results in
[`../mimic4_medmod_ihm_results.md`](../mimic4_medmod_ihm_results.md). They
train the same models with the same settings as the example scripts
`mimic4_medmod_ihm_early_fusion.py` and `mimic4_medmod_convirt.py`. They also
compute bootstrap confidence intervals, write a JSON report for every run and
save checkpoints, so that a run continues after a disconnect. The Colab
notebook [`../mimic4_medmod_ihm_colab_runs.ipynb`](../mimic4_medmod_ihm_colab_runs.ipynb)
runs all of them.

The runs need credentialed PhysioNet access to MIMIC-IV v1.0 and
MIMIC-CXR-JPG 2.0.0. Keep the data out of the code folder and out of git. The
scripts print only aggregates, such as counts, losses and scores.

| Script | What it does |
|---|---|
| `slim_events.py` | Writes the slim copy of the MIMIC-IV event tables (once) |
| `samples.py` | Builds and checks an in-hospital mortality sample set |
| `early_fusion_run.py` | Trains and tests Early Fusion |
| `pretrain_samples.py` | Builds the ConVIRT pretraining samples |
| `convirt_pretrain.py` | Pretrains ConVIRT |
| `convirt_probe.py` | Trains and tests the linear probe on a pretraining checkpoint |
| `colab_bundle.py` | Zips the code and data for Google Colab |
| `provenance.py` | Records the code version in every report and checkpoint |

Run each script from the repository root. This shows its options:
`python -m examples.medmod_runners.<script> --help`

## Data layout

Every script takes `--data-root` and finds the parts below in it. `samples.py`
can also take each part separately (`--ehr-root`, `--cxr-root`, `--cache-dir`,
`--image-dir`).

```
<data-root>/
  mimiciv/1.0/                       MIMIC-IV v1.0 (7 tables used)
    core/  patients  admissions      hosp/  labevents  diagnoses_icd  d_icd_diagnoses
    icu/   icustays  chartevents     (all .csv.gz)
  mimiciv_medmod_slim/1.0/           written by slim_events.py, see below
  mimic-cxr-jpg/2.0.0/
    mimic-cxr-2.0.0-metadata.csv.gz  (exactly this name, PyHealth reads it)
    mimic-cxr-2.0.0-chexpert.csv.gz
    files/p1x/p<subject>/s<study>/<dicom>.jpg   original images
    resized/<dicom>.jpg                         resized images, see below
  pyhealth_cache/                    PyHealth's processed copy (safe to delete, it rebuilds)
  medmod_extracted/                  MedMod's own extraction, for ConVIRT pretraining
  cohorts/pretrain_cohort.csv        the image of each pretraining stay, see below
```

PyHealth writes `mimic-cxr-2.0.0-metadata-pyhealth.csv` into the CXR folder,
so that folder must be writable.

## Preparing the data

**EHR tables.** The scripts read `mimiciv_medmod_slim/1.0`, a copy of MIMIC-IV
v1.0 with only the event rows MedMod can use. PyHealth sorts the whole event
table on disk when it builds its cache, and with the full chartevents table
this ran out of space with 25 to 30 GB free. Write the copy once (about 5
minutes):

```bash
python -m examples.medmod_runners.slim_events --data-root <data-root>
```

It keeps a chartevents or labevents row only if its item id is one of MedMod's
114 mapped item ids, its `hadm_id` and `value` are not empty, and its
`charttime` lies between 1 hour before and 49 hours after the start of an ICU
stay of the same patient. MedMod never uses the other rows, so the samples are
the same as with the full tables. The other tables are linked, not copied, and
the original files stay unchanged.

**Resized images.** The scripts read images resized the way MedFuse does
(width 512). To write them:

```python
from pyhealth.datasets.medmod.resize_cxr import paths_from_metadata, resize_cxr

root = "<data-root>/mimic-cxr-jpg/2.0.0"
print(resize_cxr(paths_from_metadata(root), f"{root}/resized"))
```

**ConVIRT pretraining data.** `medmod_extracted/` is the output of MedMod's
own extraction scripts (`mimic4extract` in https://github.com/nyuad-cai/MedMod,
run unchanged, including its phenotyping task). Pretraining reads
`medmod_extracted/root/all_stays.csv` and `medmod_extracted/phenotyping/`.
Write `cohorts/pretrain_cohort.csv` with:

```python
from pyhealth.datasets.medmod.pretraining import pretraining_cohort

cohort = pretraining_cohort("<data-root>/medmod_extracted",
                            "<data-root>/mimic-cxr-jpg/2.0.0")
cohort.to_csv("<data-root>/cohorts/pretrain_cohort.csv", index=False)
```

**MedSigLIP embeddings** (extension). `python -m pyhealth.datasets.medmod.medsiglip
--images <data-root>/mimic-cxr-jpg/2.0.0/resized --out medsiglip_448.npz`
writes them. Pass the file to the runners with `--embeddings`.

**MedFuse checkpoints** (Early Fusion). Download `uni_ehr_all` and
`cxr_rad_full` from https://github.com/nyuad-cai/MedFuse (Git LFS) and pass
them with `--ehr-weights` and `--cxr-weights`.

MedMod's item map and split lists are downloaded from MedMod's GitHub on first
use (`pyhealth.datasets.medmod.downloads`).

## Running

On Google Colab, follow the notebook
[`../mimic4_medmod_ihm_colab_runs.ipynb`](../mimic4_medmod_ihm_colab_runs.ipynb).
Its intro says which files to upload, and its "Jobs" cell has the command of
every run. On your own machine, the same commands work with
`--data-root <data-root>`:

1. `slim_events` writes the slim copy of the event tables (once).
2. `samples` builds and checks the sample set (about 15 minutes the first
   time, then cached).
3. `early_fusion_run` trains Early Fusion, once per seed.
4. `convirt_pretrain` pretrains ConVIRT, once per seed (about 6 hours on a
   GPU for the ResNet-34).
5. `convirt_probe` trains the probe on a pretraining checkpoint. The notebook
   chooses the checkpoint by validation AUROC with quick `--no-augment` probes
   on every 10th epoch.
