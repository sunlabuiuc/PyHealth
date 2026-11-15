
from multiprocessing import freeze_support

from pyhealth.datasets import eICUDataset
from pyhealth.tasks.tpc_eicu import TPCLengthOfStayEICU


def main():
    root = "eicu-crd-2.0"

    dataset = eICUDataset(
        root=root,
       tables=[
    "lab",
    "vitalperiodic",
    "vitalaperiodic",
    "intakeoutput",
    "respiratorycharting",
    "nursecharting",]
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
            print("Dynamic structure:", [
                getattr(x, "shape", type(x).__name__)
                for x in sample["dynamic"]])
            print("LOS type:", type(sample["los"]))
            print("LOS structure:", [
                getattr(x, "shape", type(x).__name__)
                for x in sample["los"]])


    print("\nIntegration test finished.")


if __name__ == "__main__":
    freeze_support()
    main()
