pyhealth.tasks.multi_view_time_series_task
==========================================

The ``multi_view_time_series_task`` module provides a standalone task for
generating three synchronized EEG views per epoch:

- Temporal view (raw signal)
- Derivative view (first-order difference)
- Frequency view (FFT magnitude)

.. autoclass:: pyhealth.tasks.multi_view_time_series_task.MultiViewTimeSeriesTask
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: pyhealth.tasks.multi_view_time_series_task.load_epoch_views

.. autofunction:: pyhealth.tasks.multi_view_time_series_task.get_view_shapes
