"""Fit processors on training samples and reuse them for test samples.

Fitting processors only on training data keeps test patients from shaping
preprocessing (vocabularies, and any statistics a processor learns). Pass the
training dataset's processors when building the test dataset: supplied
processors are reused as they are, never refitted.

Runs in a few seconds on CPU with synthetic data; no download.

Usage:
    python examples/reuse_train_processors.py
"""

from pyhealth.datasets import create_sample_dataset

INPUT_SCHEMA = {"codes": "sequence", "age": "tensor"}
OUTPUT_SCHEMA = {"label": "binary"}

train_samples = [
    {"patient_id": "p1", "codes": ["I10", "E11"], "age": [61.0], "label": 1},
    {"patient_id": "p2", "codes": ["J45"], "age": [34.0], "label": 0},
    {"patient_id": "p3", "codes": ["E11", "N18"], "age": [72.0], "label": 1},
]
test_samples = [
    # "C50" never occurs in training, so it maps to <unk>, as at deployment
    {"patient_id": "p9", "codes": ["I10", "C50"], "age": [58.0], "label": 0},
]


def main():
    train = create_sample_dataset(
        train_samples, INPUT_SCHEMA, OUTPUT_SCHEMA, dataset_name="train"
    )
    codes = train.input_processors["codes"]
    print("training vocabulary:", sorted(codes.code_vocab))

    # Reuse every fitted processor. (A field left out of input_processors would
    # get a processor fitted on the samples passed here, i.e. on test data.)
    test = create_sample_dataset(
        test_samples,
        INPUT_SCHEMA,
        OUTPUT_SCHEMA,
        input_processors=train.input_processors,
        output_processors=train.output_processors,
        dataset_name="test",
    )
    print("same codes processor:", test.input_processors["codes"] is codes)
    print("test sample:", test[0])


if __name__ == "__main__":
    main()
