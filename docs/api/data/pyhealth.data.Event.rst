pyhealth.data.Event
=========================

One basic data structure in the package. It is a simple container for a single event. 
It contains all necessary attributes for supporting various healthcare tasks.

Events without a time
---------------------

Some tables have no timestamp column, for example MIMIC-III/IV ``patients``
(demographics) and the eICU tables. Their events have ``timestamp`` set to
``None``, matching the ``null`` value in ``patient.get_events(..., return_df=True)``.
Check for it before comparing or sorting by time:

.. code-block:: python

    timed = [e for e in patient.get_events() if e.timestamp is not None]

.. note::

   In PyHealth 2.0.2 and earlier, a missing timestamp was replaced with
   ``datetime.now()``, which produced a different, invented time on every call.
   Pass ``timestamp=`` explicitly if you create events by hand and need a time.

``Event`` objects can be copied (``copy.copy`` / ``copy.deepcopy``) and pickled,
so they can be stored in task samples and sent to worker processes.

.. autoclass:: pyhealth.data.Event
    :members:
    :undoc-members:
    :show-inheritance:
