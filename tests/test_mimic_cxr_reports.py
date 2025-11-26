from pyhealth.datasets.mimic_cxr_reports import MIMICCXRReportsDataset

def main():
    ds = MIMICCXRReportsDataset(
        root=".",                 # folder containing mimic-cxr-reports.zip
        patients=["p10", "p11"],  # optional filtering
        dev_mode=True,
        limit=5,
    )

    samples = ds.get_samples()

    print("\n---------------------------------------")
    print(f"Loaded {len(samples)} samples.")
    print("---------------------------------------\n")

    for i, sample in enumerate(samples[:3]):
        print(f"Sample {i+1}:")
        print(sample)
        print()

if __name__ == "__main__":
    main()