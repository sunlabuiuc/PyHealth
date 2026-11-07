pyhealth.data.Patient
=========================

Another basic data structure in the package. A Patient is a collection of Visit for
the current patients. It contains all necessary attributes of a patient, such as
ethnicity, mortality status, gender, etc. It can support various healthcare tasks.

Event order
-----------

A patient's events are ordered by timestamp. Events that share a timestamp are
grouped by table, sorted by event type, and within a table keep the order of
their rows in the source file: all diagnoses of a MIMIC admission carry its
discharge time, for example, so they come out in ``seq_num`` (priority) order.
Dataset loading records each row's position in a ``_source_row`` column of the
cached event table, so the order is the same on every cache build. Caches
written before this column existed are rebuilt automatically.

Some rows share a ``_source_row``. A ``join`` in a dataset config that matches
one row to several rows copies the row's ``_source_row`` to every copy: the
MIMIC-CXR ``chexpert``, ``negbio`` and ``test_set_labeled`` tables, for
example, are joined to one metadata row per image. A ``preprocess_<table>``
hook that duplicates rows does the same, and rows that a ``right`` or ``outer``
join adds without a match all get ``-1``. Rows that tie on timestamp, event
type and ``_source_row`` are ordered by their other columns (for the MIMIC-CXR
labels, by image), so their order is also the same on every build.

.. autoclass:: pyhealth.data.Patient
    :members:
    :undoc-members:
    :show-inheritance:
