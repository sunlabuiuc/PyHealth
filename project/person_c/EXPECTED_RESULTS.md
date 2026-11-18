# Independent synthetic-data oracle

These numbers were calculated from the invented source records, not emitted
by the task. The CSV generator populates every required measurement table.
Five unique patients cover three eligible stays, a short stay, and a minor.

| Stay | Age | LOS (hours) | Expected inclusion | Labels |
|---|---|---|---|---|
| 101 | 18 | 12.5 | Yes: ordinary adult | hours 5–12; 7.5, 6.5, …, 0.5 |
| 102 | 60 | 4 | No: shorter than five hours | none |
| 103 | > 89 | 400.5 | Yes: long stay | hours 5–336; final value 64.5 |
| 104 | 17 | 12 | No: minor | none |
| 105 | 45 | 8 | Yes: missing admission height | hours 5–8; 3, 2, 1, 0 |

## Stay 101 selected raw hourly values

| Table / feature | Records | Expected result |
|---|---|---|
| vitalPeriodic / HR | minute 10: 80; minute 40: 100 | hour 0 median = 90 |
| vitalPeriodic / HR | minute 120: 110; minute 359: 120 | hours 2 and 5 = 110 and 120 |
| vitalPeriodic / HR | minute -60: 250; minute 900: 200 | excluded from this stay's grid |
| lab / potassium | minute 60: 4; minute 90: 6 | hour 1 median = 5 |
| lab / potassium | minute 240: 3; minute 300: 99 | hour 4 = 3; hour 5 missing (invalid 99) |
| lab / creatinine | minute 150: 1.2 | hour 2 = 1.2 |
| vitalAperiodic / systolic BP | minute 180: 130 | hour 3 = 130 |
| intakeOutput / urine | minute 65: 100; minute 95: 50 | hour 1 sum = 150 |
| respiratoryCharting / FiO2 | minute 180: 40% | hour 3 = 40 |
| nurseCharting / temperature | minute 240: 98.6 F | hour 4 = 37 C |

Static vector for 101: `[18, 0, 165, 65]`. For 103: `[90, 1, 165, 65]`.
For 105, height remains NaN until the training-fitted processor imputes it.

## First six hours, before normalization

- HR raw: `[90, NaN, 110, NaN, NaN, 120]`.
- HR forward-filled: `[90, 90, 110, 110, 110, 120]`.
- HR decay: `[1, .75, 1, .75, .375, 1]`.
- Potassium raw: `[NaN, 5, NaN, NaN, 3, NaN]`.
- Potassium decay: `[0, 1, .75, .375, 1, .75]`.

Decay uses the authors' formula: zero before first measurement; one on a
measurement; `1 / (gap_hours * 4/3)` after it. Removing all measurements gives
all-zero decay. Imputation is applied AFTER removal. Initial missing feature
values use training statistics, or zero if never observed in training.

Source for decay only:
https://github.com/EmmaRocheteau/TPC-LoS-prediction/blob/master/eICU_preprocessing/timeseries.py

Additional tiny unit fixtures cover exactly five hours, unknown age/sex,
invalid height/weight, no measurements, two stays of the same patient,
future-observation invariance and zero/10%/20%/100% removal.

Timing correction (November 17 audit): array row b represents the interval
[b,b+1) and is timestamped b+1, when that bin is complete. A prediction at
hour 5 uses only bins 0–4. The 336-hour cap yields 336 rows, not 337.
The oracle's hour/bin numbers identify array rows, not availability timestamps.
Exact-boundary observations at minute 300 enter bin 5, available at hour 6.
This conservative floor-bin convention differs from the paper's right-closed
resampling, but prevents sub-hour look-ahead.
