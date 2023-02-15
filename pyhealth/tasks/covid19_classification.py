def covid19_classification_fn(patient):
    """Processes a single patient for the COVID-19 classification task."""
    return patient


if __name__ == "__main__":
    from pyhealth.datasets import COVID19XRayDataset

    base_dataset = COVID19XRayDataset(
        root="/srv/local/data/zw12/raw_data/covid19-radiography-database/COVID-19_Radiography_Dataset",
    )

    sample_dataset = base_dataset.set_task(covid19_classification_fn)
    print(sample_dataset.samples[0])
    print(sample_dataset[0])
    print(sample_dataset.stat())
