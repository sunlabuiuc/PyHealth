import os
import numpy as np
from glob import glob
from pyhealth.datasets.base_dataset import BaseDataset

class DailySportsActivities(BaseDataset):
    """
    UCI Daily & Sports Activities dataset.
    - Reads raw TXT files under root/a01…a19/p1…p8/
    - Stacks into (n_samples, 1125, 5)
    - Split participants 1–6 → train; 7–8 → test
    """

    domain_names = ['T', 'RA', 'LA', 'RL', 'LL']
    domain_indices = {
        'T':  list(range(0, 9)),
        'RA': list(range(9, 18)),
        'LA': list(range(18, 27)),
        'RL': list(range(27, 36)),
        'LL': list(range(36, 45)),
    }

    def __init__(self, root: str, split: str = "train"):
        # Try BaseDataset init; if no YAML config, fall back
        try:
            super().__init__(root, split)
        except Exception:
            self.root = root
            self.split = split
        # Load data
        self.data, self.labels = self._load_and_preprocess()

    def _load_and_preprocess(self):
        train_x, train_y, test_x, test_y = [], [], [], []
        for act_id in range(1, 20):
            act_dir = f"a{act_id:02d}"
            for pid in range(1, 9):
                pat = os.path.join(self.root, act_dir, f"p{pid}", "*.txt")
                for fp in sorted(glob(pat)):
                    arr = np.loadtxt(fp, delimiter=",")
                    if arr.shape != (125, 45):
                        continue
                    # split 45→ five 9-col blocks
                    blocks = [arr[:, self.domain_indices[d]] for d in self.domain_names]
                    stack  = np.stack(blocks, axis=-1)      # (125,9,5)
                    flat   = stack.reshape(125*9, 5)       # (1125,5)
                    if pid <= 6:
                        train_x.append(flat); train_y.append(act_id-1)
                    else:
                        test_x.append(flat);  test_y.append(act_id-1)

        X_tr, y_tr = np.array(train_x), np.array(train_y)
        X_te, y_te = np.array(test_x),  np.array(test_y)
        X, y = (X_tr, y_tr) if self.split=="train" else (X_te, y_te)

        # (Optional) normalize to [-1,1]:
        # X = 2*(X - X.min())/(X.max()-X.min()) - 1

        return X, y
