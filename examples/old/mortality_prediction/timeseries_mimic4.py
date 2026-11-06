from pyhealth.datasets import MIMIC4Dataset
if __name__ == "__main__":
    dataset = MIMIC4Dataset(
        ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.2/",
        ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"],
        dev=True,
        cache_dir="../benchmark_cache/mimic4_ehr/"
    )

    from pyhealth.tasks import InHospitalMortalityMIMIC4

    task = InHospitalMortalityMIMIC4()
    samples = dataset.set_task(task, num_workers=2)

    from pyhealth.datasets import split_by_sample


    train_dataset, val_dataset, test_dataset = split_by_sample(
        dataset=samples, ratios=[0.7, 0.1, 0.2]
    )


