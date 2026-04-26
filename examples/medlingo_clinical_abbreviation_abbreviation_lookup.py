"""
This example demonstrates an evaluation of the Clinical Abbreviation Task using the MedLingo dataset. 
We will perform ablation studies to understand the impact of different input modifications on the model's performance.

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

def main() -> None:
    dataset = MedLingoDataset(root="test-resources")
    records = dataset.process()

    samples = []
    for record in records:
        for sample in record["medlingo"]:
            samples.append(sample)

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