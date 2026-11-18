pyhealth.datasets.JichiDRDataset
================================

Overview
--------

Jichi Medical University diabetic retinopathy dataset: 9,939 fundus
photographs of 2,740 patients, each graded on the modified Davis scale (no DR,
simple DR, pre-proliferative DR, proliferative DR), with a single-image grade
and a grade read with the patient's other images. Open access under CC BY 4.0.
``root`` is the extracted ``dmr/`` folder (``list.csv`` plus the JPEGs) or its
parent. The patient id is taken from the file-name prefix.

Source: `Davis Grading of One and Concatenated Figures (figshare) <https://figshare.com/articles/dataset/4879853>`_

API Reference
-------------

.. autoclass:: pyhealth.datasets.JichiDRDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.jichi.parse_jichi_filename
