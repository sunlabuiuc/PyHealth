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

One exception: a ``join`` in a dataset config that matches several rows to one
row copies that row's ``_source_row`` to every match, so the order among the
copies can differ between builds. The bundled configs only join one row per
key.

.. autoclass:: pyhealth.data.Patient
    :members:
    :undoc-members:
    :show-inheritance:
