"""
This example demonstrates an evaluation of the Clinical Abbreviation Task using the MedLingo dataset. 

All samples used in this example are synthetic and defined inline to ensure reproducibility and to avoid
reliance on external or real clinical datasets.

We perform ablation studies to understand the impact of different input modifications on model performance.

The ablation studies include:
1. Base abbreviation-only input. 
2. Ablation with Lowercase formatting. 
3. Ablation with short clinical context.
4. Ablation with Noisy formatting. 

Paper:
    Diagnosing Our Datasets: How Does My Language Model Learn Clinical Information?
    https://arxiv.org/abs/2505.15024

"""

from pyhealth.datasets.medlingo import MedLingoDataset
from pyhealth.tasks.clinical_abbreviation import ClinicalAbbreviationTask

SYNTHETIC_MEDLINGO_SAMPLES = [
    {
        "abbr": "SOB",
        "context": "Patient presents with SOB.",
        "label": "shortness of breath",
        "source": "synthetic_demo",
    },
    {
        "abbr": "BP",
        "context": "BP remained stable overnight.",
        "label": "blood pressure",
        "source": "synthetic_demo",
    },
    {
        "abbr": "HTN",
        "context": "History of HTN.",
        "label": "hypertension",
        "source": "synthetic_demo",
    },
    {
        "abbr": "CHF",
        "context": "Known CHF with fluid overload.",
        "label": "congestive heart failure",
        "source": "synthetic_demo",
    },
    {
        "abbr": "DM",
        "context": "History of DM with elevated glucose.",
        "label": "diabetes mellitus",
        "source": "synthetic_demo",
    },
]

def main() -> None:
    dataset = MedLingoDataset(samples=SYNTHETIC_MEDLINGO_SAMPLES)
    records = dataset.process()

    samples = []
    for record in records:
        for s in record["medlingo"]:
            samples.append(s)

    print("=== Base Results: Abbreviation-Only ===")
    base_task = ClinicalAbbreviationTask(use_context=False)
    for sample in samples:
        print(base_task(sample))

    print("\n=== Ablation 1: Lowercase Input ===")
    for sample in samples:
        modified = {
            **sample,
            "abbr": sample["abbr"].lower(),
        }
        print(base_task(modified))

    print("\n=== Ablation 2: Short Clinical Context ===")
    context_task = ClinicalAbbreviationTask(use_context=True)
    for sample in samples:
        print(context_task(sample))

    print("\n=== Ablation 3: Noisy Formatting ===")
    noise_variants = ["!!!", "???", "..."]

    for sample in samples:
        for noise in noise_variants:
            noisy = {
                **sample,
                "abbr": sample["abbr"] + noise,
            }
            print(base_task(noisy))


if __name__ == "__main__":
    main()