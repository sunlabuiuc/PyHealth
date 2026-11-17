pyhealth.models.MotionAge
=========================

MotionAge (Song, Yin, et al., `MotionAge: A Deep Learning Framework for
Biological Age Prediction from Wearable Activity
<https://openreview.net/pdf?id=xGw1o9acWa>`_) predicts 5-year mortality risk
from wearable activity and converts it to a biological age. The architecture
and mapping follow the authors' code
(https://github.com/jackie-jiaqi-yin/MotionAge, MIT License).

:class:`~pyhealth.models.MotionAge` runs a GRU (two layers by default) over
the 5-minute epochs, using intensity plus hour and day embeddings, and
averages the hidden states over worn epochs only. Covariates, if present, go
through an MLP and are combined by late or residual fusion. Default sizes
follow the authors' GRU configs. Use it with
:class:`~pyhealth.tasks.NHANESMortalityPrediction`.

:class:`~pyhealth.models.MotionAgeMapping` maps predicted risk to MotionAge.
For each sex it fits ``logit(risk) = alpha + beta * age`` on training
participants (median risk per age, weighted by count) and inverts the line.
MotionAge minus age is the age acceleration.

.. autoclass:: pyhealth.models.MotionAge
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.MotionAgeMapping
    :members:
    :undoc-members:
    :show-inheritance:
