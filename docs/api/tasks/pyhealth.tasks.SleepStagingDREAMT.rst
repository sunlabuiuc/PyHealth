pyhealth.tasks.SleepStagingDREAMT
=======================================

.. autoclass:: pyhealth.tasks.sleep_staging_dreamt.SleepStagingDREAMT
    :members:
    :undoc-members:
    :show-inheritance:

Usage
-----

The default configuration processes BVP recordings and converts each
30-second epoch into an IBI sequence:

.. code-block:: python

    from pyhealth.tasks import SleepStagingDREAMT

    task = SleepStagingDREAMT()

To use precomputed IBI values instead, set ``signal_type="IBI"``:

.. code-block:: python

    task = SleepStagingDREAMT(signal_type="IBI")

The input column names can also be customized:

.. code-block:: python

    task = SleepStagingDREAMT(
        signal_type="IBI",
        ibi_column="PulseInterval",
        stage_column="Stage",
    )

The task maps sleep stages to three classes: Wake (0), NREM (1), and
REM (2). Its default output signal shape is ``(1, 1920)``.


For temporal models such as TCN, set ``sequence_first=True`` to preserve
the time dimension as the first signal dimension. This changes the output
shape to ``(1920, 1)`` for the default 64 Hz sampling rate and 30-second
epochs:

.. code-block:: python

    task = SleepStagingDREAMT(sequence_first=True)
