import os
import pickle
import numpy as np

for sid in range(2, 17):
    if sid in (1, 12):
        continue
    subject_id = f"S{sid}"
    subject_dir = os.path.join("test_data/wesad", subject_id)
    os.makedirs(subject_dir, exist_ok=True)

    # Fake sensor data
    signal = {
        "chest": {
            "ACC": np.random.rand(10).tolist(),
            "ECG": np.random.rand(10).tolist(),
            "EMG": np.random.rand(10).tolist(),
        },
        "wrist": {
            "ACC": np.random.rand(10).tolist(),
            "EDA": np.random.rand(10).tolist(),
            "TEMP": np.random.rand(10).tolist(),
        }
    }
    labels = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]

    data = {
        "subject": subject_id,
        "signal": signal,
        "label": labels
    }

    with open(os.path.join(subject_dir, f"{subject_id}.pkl"), "wb") as f:
        pickle.dump(data, f)