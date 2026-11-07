"""Run: python examples/keyclass_synthetic.py [--aggregation snorkel]."""

import argparse

import numpy as np
import torch
from sklearn.feature_extraction.text import HashingVectorizer

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import KeyClass
from pyhealth.weak_supervision import KeywordLabeler, aggregate_votes, vote_diagnostics


class SyntheticEncoder:
    """Offline lexical mock, NOT a clinical semantic encoder."""

    def encode(self, texts):
        return (
            HashingVectorizer(n_features=64, alternate_sign=False)
            .transform(texts)
            .toarray()
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregation", choices=["majority", "snorkel"], default="majority"
    )
    args = parser.parse_args()
    torch.manual_seed(0)
    descriptions = {
        "respiratory": "pneumonia asthma lung",
        "circulatory": "heart cardiac hypertension",
        "renal": "kidney renal dialysis",
    }
    # Teammate boundary: rows need text; patient/admission IDs stay with rows.
    # Gold labels are deliberately absent from all weak-supervision operations.
    rows = [
        {"patient_id": str(i), "text": text}
        for i, text in enumerate(
            [
                "pneumonia lung",
                "asthma lung",
                "heart cardiac",
                "heart hypertension",
                "kidney renal",
                "renal dialysis",
                "pneumonia heart",
                "kidney heart",
                "routine observation",
            ]
        )
    ]
    texts = [r["text"] for r in rows]
    encoder = SyntheticEncoder()
    labeler = KeywordLabeler(encoder, top_k=3, min_df=1).fit(texts, descriptions)
    votes = labeler.transform(texts)
    targets, covered = aggregate_votes(
        votes, len(descriptions), method=args.aggregation
    )
    embeddings = encoder.encode(texts)
    samples = [
        {
            "patient_id": rows[i]["patient_id"],
            "embedding": embeddings[i].tolist(),
            "weak_labels": targets[i].tolist(),
        }
        for i in np.flatnonzero(covered)
    ]
    dataset = create_sample_dataset(
        samples=samples,
        input_schema={"embedding": "tensor"},
        output_schema={"weak_labels": "tensor"},
    )
    model = KeyClass(dataset, embeddings.shape[1], labeler.category_ids_)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)
    loader = get_dataloader(dataset, batch_size=4, shuffle=True)
    for _ in range(2):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            result = model(**batch)
            result["loss"].backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        predictions = model(embedding=encoder.encode(["pneumonia heart"]))["y_prob"]
    print("Category order:", labeler.category_ids_)
    print("LF diagnostics:", vote_diagnostics(votes, len(descriptions)))
    print("Training loss:", float(result["loss"].detach()))
    print("Synthetic prediction (not a quality evaluation):", predictions.tolist())


if __name__ == "__main__":
    main()
