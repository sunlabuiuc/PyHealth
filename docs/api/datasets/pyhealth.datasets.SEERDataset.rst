SEERDataset
===========

This dataset reads a local SEER\*Stat Case Listing exported as comma-delimited
text with variable names included. Save the listing as ``seer_export.txt`` in
the dataset root using Windows-1252 encoding or UTF-8 with a byte-order mark.
The export dictionary can help verify the column names.
Access to real records requires authorization from the
`NCI SEER program <https://seer.cancer.gov/data/access.html>`_.

Each tumor is a ``tumor`` event. Patient ID groups events, and record number
distinguishes tumors within a patient. Diagnosis year is an integer attribute.
Diagnosis-time fields retain their formatted values, including leading zeros
in codes. Blank diagnosis-time values are represented by empty strings.
The loader checks the exported age recode, ICD-O-3 histology code, and AJCC
stage formats and accepts the export's missing and unknown categories.
Survival months is a nullable integer, and cause of death is normalized to
``cancer_death``, ``alive_or_other_cause``, ``unknown``, or
``not_applicable``. Use these outcome fields to build task labels only.

The expected input headers and normalized field names are defined in
``pyhealth/datasets/seer.py`` and ``pyhealth/datasets/configs/seer.yaml``.
The export may include additional fields. Store the source and cache in
locations approved for the data.

Save and run this Python script from a PyHealth checkout:

.. code-block:: python

    from tempfile import TemporaryDirectory
    from pyhealth.datasets import SEERDataset

    if __name__ == "__main__":
        with TemporaryDirectory() as cache:
            dataset = SEERDataset("tests/fixtures/seer", cache_dir=cache)
            print(len(dataset.unique_patient_ids))

.. autoclass:: pyhealth.datasets.SEERDataset
    :members:
