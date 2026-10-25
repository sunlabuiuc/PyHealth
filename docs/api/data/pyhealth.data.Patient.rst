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

Joins in a dataset config are the exception. A ``join`` that matches several
rows to one row copies that row's ``_source_row`` to every match, and rows that
a ``right`` or ``outer`` join adds without a match get ``-1``. The order among
such rows can differ between builds. The bundled configs use ``left`` and
``inner`` joins that look up one row per key.

.. autoclass:: pyhealth.data.Patient
    :members:
    :undoc-members:
    :show-inheritance:
