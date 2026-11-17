pyhealth.metrics.motionage
==========================

Participant-level evaluation for MotionAge (Song, Yin, et al.). The helpers
average a participant's window probabilities into one mortality risk, collect
the age and sex that :class:`~pyhealth.tasks.NHANESMortalityPrediction` passes
through, compute age acceleration (MotionAge minus age), the paper's C-index
C(M, X) of a raw predictor against the binary mortality label, and the AUROC
of logistic regressions fitted on training participants and scored on test
participants. These functions are not exported from ``pyhealth.metrics``;
import them from ``pyhealth.metrics.motionage``.

.. code-block:: python

    from pyhealth.metrics.motionage import (
        age_acceleration,
        aggregate_window_predictions,
        evaluate_feature_sets,
        label_concordance,
        participant_fields,
    )
    from pyhealth.models.motionage import MotionAgeMapping

    y_true, y_prob, _, ids = trainer.inference(loader, return_patient_ids=True)
    table = aggregate_window_predictions(ids, y_prob, y_true).merge(
        participant_fields(samples), on="patient_id"
    )
    mapping = MotionAgeMapping().fit(train["risk"], train["age"], train["sex"])
    table["motion_age"] = mapping.transform(table["risk"], table["sex"])
    table["age_acceleration"] = age_acceleration(table["motion_age"], table["age"])
    c_index = label_concordance(table["label"], table["motion_age"])
    # evaluate_feature_sets(train, test, {"age": ["age", "sex"], ...})

.. currentmodule:: pyhealth.metrics.motionage

.. autofunction:: aggregate_window_predictions

.. autofunction:: participant_fields

.. autofunction:: age_acceleration

.. autofunction:: label_concordance

.. autofunction:: evaluate_feature_sets
