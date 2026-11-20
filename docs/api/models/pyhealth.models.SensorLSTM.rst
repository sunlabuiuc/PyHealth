pyhealth.models.SensorLSTM
==========================

An LSTM classifier over one raw multichannel sensor stream, used as the base
network of the adaptive transfer framework of Zhang et al. (2024). Its
``input_key`` selects which stream is read, so one set of weights can be
pretrained on source sensors and fine-tuned on the target sensor (see
:mod:`pyhealth.models.sensor_transfer`).

.. autoclass:: pyhealth.models.SensorLSTM
    :members:
    :undoc-members:
    :show-inheritance:
