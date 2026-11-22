"""
Contributors: Manling Tan, Yining Jin, Xiao Luo
NetID: manling4, yining34, xiaoluo6

Paper Title:
Temporal Pointwise Convolutional Networks for
Length of Stay Prediction in the Intensive Care Unit

Paper Authors:
Emma Rocheteau, Pietro Liò, and Stephanie Hyland (2021)

Paper URL:
https://arxiv.org/abs/2007.09483
"""


"""Integration test for TPC hourly ICU length-of-stay prediction.

Loads real eICU data and inspects generated task samples
under different missing-data rates.
"""

from multiprocessing import freeze_support

from pyhealth.datasets import eICUDataset
from pyhealth.tasks.tpc_eicu import TPCLengthOfStayEICU


def main() -> None:
    """Test TPC task processing on real eICU data.

    Loads the eICU dataset and generates task samples using
    missing rates of 0%, 10%, and 20%.

    Prints the number of generated samples and the structure
    of dynamic features and LOS labels for each missing rate.
    """
    root = "eicu-crd-2.0"

    dataset = eICUDataset(
        root=root,
        tables=[
            "lab",
            "vitalperiodic",
            "vitalaperiodic",
            "intakeoutput",
            "respiratorycharting",
            "nursecharting",
        ],
    )

    print("Dataset loaded successfully.")

    for rate in [0.0, 0.1, 0.2]:
        print(f"\nTesting missing rate: {rate:.0%}")

        task = TPCLengthOfStayEICU(
            missing_rate=rate,
            seed=42,
        )

        samples = dataset.set_task(
            task=task,
            num_workers=1,
        )

        print("Number of samples:", len(samples))

        if len(samples) > 0:
            sample = samples[0]

            print("Dynamic type:", type(sample["dynamic"]))
            print(
                "Dynamic structure:",
                [getattr(x, "shape", type(x).__name__) for x in sample["dynamic"]],
            )

            print("LOS type:", type(sample["los"]))
            print(
                "LOS structure:",
                [getattr(x, "shape", type(x).__name__) for x in sample["los"]],
            )

    print("\nIntegration test finished.")


if __name__ == "__main__":
    freeze_support()
    main()
