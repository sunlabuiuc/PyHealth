pyhealth.datasets.DSADataset
============================

The Daily and Sports Activities (DSA) dataset: 19 daily and sports activities
performed by 8 subjects and recorded by five Xsens motion-sensor units (torso,
right arm, left arm, right leg, left leg), each with 9 channels at 25 Hz. Every
activity is split into 5-second segments of 125 samples. See the
`UCI page <https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities>`_
for more information.

The dataset is used by Zhang et al. (2024), *Daily Physical Activity Monitoring:
Adaptive Learning from Multi-source Motion Sensor Data* (CHIL 2024). Use it with
:class:`~pyhealth.tasks.DSAActivityClassification`. The five sensor streams of a
segment are synchronized, which is what the source-to-target transfer of the
paper relies on.

.. autoclass:: pyhealth.datasets.DSADataset
    :members:
    :undoc-members:
    :show-inheritance:

Helper functions
----------------

.. autofunction:: pyhealth.datasets.dsa.build_dsa_metadata

.. autofunction:: pyhealth.datasets.dsa.load_dsa_segment

.. autofunction:: pyhealth.datasets.dsa.split_sensors

.. autofunction:: pyhealth.datasets.dsa.compute_channel_ranges
