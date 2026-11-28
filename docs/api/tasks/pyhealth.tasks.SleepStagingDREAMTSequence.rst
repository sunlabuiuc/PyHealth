pyhealth.tasks.SleepStagingDREAMTSequence
=========================================

.. autoclass:: pyhealth.tasks.sleep_staging_dreamt_sequence.SleepStagingDREAMTSequence
    :members:
    :undoc-members:
    :show-inheritance:

Overview
--------

This task prepares DREAMT wrist-worn BVP recordings for sequence-level
sleep staging with WatchSleepNet.

The task processes BVP recordings sampled at 100 Hz, extracts inter-beat
intervals (IBIs), and downsamples the IBI signal to 25 Hz. Each 30-second
epoch contains 750 IBI samples.

Sleep stages are mapped to three classes:

* Wake (0)
* NREM (1), combining N1, N2, and N3
* REM (2)

The output contains a chronological sequence of epoch-level IBI signals
and their corresponding sleep-stage labels for each participant. Invalid
or inconsistent epochs are excluded.

Usage
-----

.. code-block:: python

    from pyhealth.tasks import SleepStagingDREAMTSequence

    task = SleepStagingDREAMTSequence()

The default configuration expects the DREAMT 100 Hz BVP column named
``BVP`` and the sleep-stage column named ``Sleep_Stage``. NeuroKit2 is
required for IBI extraction and can be installed with:

.. code-block:: bash

    pip install pyhealth[dreamt]

Each participant produces a variable-length sequence with 750 samples
per epoch and one label per valid epoch.

Output structure
----------------

For a participant with N valid sleep epochs, the signal has shape
``(N, 750)``. The task returns chronological sleep-stage labels and
their associated timestamps.

N1, N2, and N3 are combined into the NREM class. Epoch construction
uses contiguous runs of mapped sleep-stage labels. Incomplete epochs
and unsupported sleep stages are excluded.

Elapsed recording offsets are retained so that gaps between
valid epochs can be identified when constructing training sequences.

Context-aware IBI extraction
----------------------------

By default, ``context_seconds=0`` preserves independent epochwise
IBI extraction and the original sequence preprocessing behavior.

An optional experimental extension uses surrounding BVP samples
to reduce boundary-related IBI gaps:

.. code-block:: python

    from pyhealth.tasks import SleepStagingDREAMTSequence

    task = SleepStagingDREAMTSequence(context_seconds=5)

With five seconds of context, IBI extraction uses up to five
additional seconds of BVP on each side of the target epoch.
The extracted IBI signal is cropped back to the original
30-second epoch and downsampled to 750 samples.

Context is clipped at recording boundaries. Valid, continuous
numeric ``TIMESTAMP`` values are required for context extraction.
Missing timestamps, discontinuities, nonfinite context signals,
or invalid extraction results cause the task to retain the
original epochwise IBI result.

The extension preserves epoch labels, elapsed recording offsets,
and output dimensions. Because it can use future BVP samples,
it is intended for retrospective rather than strictly online
sleep staging.

Reduced IBI gaps do not necessarily imply improved classification
performance. Predictive effectiveness should be evaluated using
participant-disjoint experiments.


Data requirements
-----------------

The sequence task requires the DREAMT 100 Hz recording files.
The DREAMT dataset is available from PhysioNet:

https://physionet.org/content/dreamt/

Users must obtain the required dataset access and comply with the
applicable data use agreement. Restricted participant recordings
should not be included in public repositories or examples.
