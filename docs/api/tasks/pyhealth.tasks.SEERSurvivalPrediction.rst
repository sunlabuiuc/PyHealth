SEERSurvivalPrediction
======================

The task produces one binary sample per eligible tumor from
``SEERDataset``. The default horizon is 12 months. A cancer death before
the horizon receives label 0 and a record with observed survival at or
beyond the horizon receives label 1. The task drops early censoring, other
deaths before the horizon, missing duration, and unknown or inapplicable
cause of death. Patient and tumor identifiers are sample metadata.

The inputs are an explicit list of diagnosis-time fields. It includes
diagnosis year, primary site, sex, age group, histology, grade, laterality,
marital status, AJCC stage group, tumor size, extension, lymph nodes, and
metastases. The site cohort filter, survival outcomes, follow-up fields,
and treatment variables are outside that list.

Use ``encoding="one_hot"`` to one-hot encode categorical values. Diagnosis
year and ordinary tumor sizes (1–988 mm) are standardized. Blank values,
previously unseen categories, and observed special size codes have distinct
indicator columns within their field. A missing or special size has zero in
its standardized measurement column.

Use ``encoding="numeric"`` for the standardized numeric baseline. The
processor uses the numeric diagnosis year, size, and verified exported codes
for primary site, histology, extension, lymph nodes, and metastases. It
assigns zero-based ordinal codes in sorted order to formatted categories
observed in training. Blank values use -1 and unseen formatted categories are
imputed to the training mean. These ordinal codes approximate the paper's
numeric input representation on a newer formatted export. Exported codes are
used as numbers, so a special code such as tumor size 999 (unknown) is
standardized like a measurement; one-hot encoding keeps it separate.

Pass ``split=`` to ``dataset.set_task`` so PyHealth fits vocabulary and
scaling on training patients only. Use ``include_extent=False`` with the same split to
remove stage, tumor size, extension, lymph nodes, and metastases together.
The fitted processor's ``field_slices`` maps source fields to vector columns.

.. code-block:: python

    from pyhealth.datasets import PatientSplit, SEERDataset
    from pyhealth.tasks import SEERSurvivalPrediction

    if __name__ == "__main__":
        dataset = SEERDataset("local_data", cache_dir="local_data/cache")
        split = PatientSplit(ratios=(0.8, 0.1, 0.1), seed=42)
        task = SEERSurvivalPrediction(encoding="one_hot")
        train, validation, test = dataset.set_task(task, split=split)

.. autoclass:: pyhealth.tasks.SEERSurvivalPrediction
    :members:
