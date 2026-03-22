import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pyhealth.datasets import eICUDataset

def main():
    root = "/home/medukonis/Documents/Illinois/Spring_2026/CS598_Deep_Learning_For_Healthcare/Project/Datasets/eicu-collaborative-research-database-2.0"

    dataset = eICUDataset(
        root=root,
        tables=["patient", "lab"],
        dev=True,
    )

    patient_id = next(iter(dataset.patients))
    patient = dataset.patients[patient_id]

    print("patient type:", type(patient))
    print("patient dir:", [x for x in dir(patient) if not x.startswith("_")])
    print("patient dict keys:", list(vars(patient).keys()))

    visits = getattr(patient, "visits", None)
    visits_dict = getattr(patient, "visits_dict", None)

    print("has visits:", visits is not None)
    print("has visits_dict:", visits_dict is not None)

    if visits is not None:
        print("visits type:", type(visits))
        try:
            print("visits len:", len(visits))
        except Exception:
            pass

    if visits_dict is not None:
        print("visits_dict type:", type(visits_dict))
        print("visits_dict keys sample:", list(visits_dict.keys())[:5])

if __name__ == "__main__":
    main()
