ECGQADataset
============

.. currentmodule:: pyhealth.datasets

.. autoclass:: ECGQADataset
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

``ECGQADataset`` loads the ECG-QA PTB-XL question-answering benchmark into
PyHealth's base dataset format. Each QA item is converted into an event-style
row with question text, answer metadata, template metadata, and ECG identifier
references.

Expected directory structure
----------------------------

The dataset expects a root directory with the following structure::

    ecgqa/ptbxl/
        answers.csv
        answers_for_each_template.csv
        train_ecgs.tsv
        valid_ecgs.tsv
        test_ecgs.tsv
        paraphrased/
            train/*.json
            valid/*.json
            test/*.json
        template/
            train/*.json
            valid/*.json
            test/*.json

Main options
------------

- ``split``: one of ``train``, ``valid``, or ``test``
- ``question_source``: one of ``paraphrased`` or ``template``
- ``question_types``: optional filter over question type
- ``attribute_types``: optional filter over attribute type
- ``single_ecg_only``: keep only samples tied to exactly one ECG