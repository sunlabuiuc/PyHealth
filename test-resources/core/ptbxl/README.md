# Synthetic PTB-XL-shaped fixture for unit tests (NOT real PhysioNet data).
#
# Contents:
# - ptbxl_database.csv / scp_statements.csv: tiny CSV stubs (committed)
# - records100 / records500: synthesized at test time by
#   tests/core/test_ptbxl.py::_materialize_fixture (12 leads x 50 samples
#   at 100 Hz; 12 x 250 at 500 Hz — unequal dims so a missing transpose
#   fails the shape assertion). Waveform binaries are not committed.
#
# Edge-case coverage:
# - ecg_id=2: age=300 (HIPAA censored ≥90); likelihood 0 (SR)
# - ecg_id=3: missing age + non-empty non-diagnostic dict (PACE-only)
# - ecg_id=4: true multi-label (IMI+LVH → MI and HYP)
# - ecg_id=5: empty scp_codes dict {}
# - ecg_id=1 and 2 share patient_id (patient-level fold leakage checks)
