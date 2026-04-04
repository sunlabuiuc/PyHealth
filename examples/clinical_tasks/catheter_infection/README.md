# Catheter Infection MIMIC-IV Examples

These scripts train baseline models for catheter-associated infection prediction on MIMIC-IV.

## Shared Config

- `MIMIC4_ROOT=/shared/rsaas/physionet.org/files/mimiciv/2.2`
- `CACHE_DIR=/shared/eng/pyhealth_agent/cache`
- Metrics include `pr_auc` and `roc_auc`
- Checkpoint monitor is `precision`

## Available Scripts

- `rnn.py`
- `retain.py`
- `transformer.py`
- `adacare.py`
- `concare.py`
- `stagenet.py`

## Run

```bash
cd /home/johnwu3/projects/PyHealth_Branch_Testing/PyHealth
python examples/clinical_tasks/catheter_infection/rnn.py --gpu_id 0
python examples/clinical_tasks/catheter_infection/stagenet.py --gpu_id 0
```

All scripts print final test metrics at the end.
