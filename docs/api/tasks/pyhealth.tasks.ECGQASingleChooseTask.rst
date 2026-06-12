ECGQASingleChooseTask
=====================

.. currentmodule:: pyhealth.tasks

.. autoclass:: ECGQASingleChooseTask
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

``ECGQASingleChooseTask`` converts ECG-QA event rows into sample-level
multiclass classification examples for single-choose question answering.

Input / output
--------------

Input schema:

- ``question`` -> ``sequence``

Output schema:

- ``label`` -> ``multiclass``

Behavior
--------

The task supports:

- filtering by ``question_types``
- optionally requiring exactly one ECG per example
- optionally removing examples whose label is ``none``