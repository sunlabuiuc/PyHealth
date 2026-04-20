from pyhealth.datasets import MIMIC3CirculatoryFailureDataset
from pyhealth.tasks import CirculatoryFailurePredictionTask
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score


def samples_to_df(samples):
    """Convert samples to DataFrame"""
    df = pd.DataFrame(
        [
            {
                "patient_id": s["patient_id"],
                "icustay_id": s["icustay_id"],
                "gender": s["gender"],
                "timestamp": s["timestamp"],
                "map": s["features"]["map"],
                "label": s["label"],
            }
            for s in samples
        ]
    )
    return df


def train_model(df):
    """Train a simple baseline model"""

    X = df[["map"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))

    return model


def main():
    dataset = MIMIC3CirculatoryFailureDataset(
        root="/Users/bella/Desktop/UIUC MCS/CS598/mimic_test"
    )
    task = CirculatoryFailurePredictionTask()

    print("Building dataset...")
    samples = dataset.set_task(task, max_patients=20)

    print(f"Total samples: {len(samples)}")

    df = samples_to_df(samples)
    print(df.head())

    print("\nTraining model...")
    model = train_model(df)


if __name__ == "__main__":
    main()