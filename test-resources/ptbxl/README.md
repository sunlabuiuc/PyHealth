# Synthetic PTB-XL-shaped fixture for unit tests (NOT real PhysioNet data).
#
# Contents are programmatically generated:
# - ptbxl_database.csv / scp_statements.csv: tiny CSV stubs
# - records100/*.hea+.dat and records500/*.hea+.dat: synthetic WFDB
#   (12 leads x 50 samples at 100 Hz; 12 x 250 at 500 Hz — unequal dims
#   so a missing transpose fails the shape assertion)
#
# Edge-case coverage:
# - ecg_id=2: age=300 (HIPAA censored ≥90); likelihood 0 (SR)
# - ecg_id=3: missing age + non-empty non-diagnostic dict (PACE-only)
# - ecg_id=4: true multi-label (IMI+LVH → MI and HYP)
# - ecg_id=5: empty scp_codes dict {}
# - ecg_id=1 and 2 share patient_id (patient-level fold leakage checks)
