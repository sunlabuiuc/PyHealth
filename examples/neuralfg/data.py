import pandas as pd

from pyhealth.datasets import create_sample_dataset


DATA_URL = (
    "https://raw.githubusercontent.com/chl8856/DeepHit/"
    "master/sample%20data/SYNTHETIC/synthetic_comprisk.csv"
)


def load_data():
    """Load the DeepHit synthetic competing-risk dataset."""

    df = pd.read_csv(DATA_URL)

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Event counts:")
    print(df["label"].value_counts().sort_index())

    return df


def prepare_pyhealth_samples(df, max_samples=None):
    """Convert DeepHit rows into PyHealth samples."""

    if max_samples is not None:
        df = df.sample(
            n=min(max_samples, len(df)),
            random_state=42,
        ).copy()

        print("\nSubset event counts:")
        print(df["label"].value_counts().sort_index())

    excluded_columns = {
        "time", "label", "true_time", "true_label"
    }

    feature_columns = [
        col for col in df.columns
        if col not in excluded_columns
    ]

    print("Number of features:", len(feature_columns))

    samples = []

    for idx, row in df.iterrows():
        samples.append({
            "patient_id": f"synthetic-{idx}",
            "features": row[feature_columns].astype(
                "float32"
            ).tolist(),
            "duration": [float(row["time"])],
            "event": [int(row["label"])],
        })

    return samples


def create_dataset(samples):
    """Create a PyHealth SampleDataset."""

    return create_sample_dataset(
        samples=samples,
        input_schema={"features": "tensor"},
        output_schema={
            "duration": "tensor",
            "event": "tensor",
        },
        dataset_name="deephit_synthetic",
    )