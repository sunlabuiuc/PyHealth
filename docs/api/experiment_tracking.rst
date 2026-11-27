Optional experiment tracking
============================

``record_experiment`` writes caller-supplied run parameters and aggregate
metrics to a local JSON manifest, or to an explicitly configured MLflow
tracking URI. It is independent of :class:`~pyhealth.trainer.Trainer` and can
be used with custom training loops. It does not automatically collect metrics,
model weights, patient data or interrupted runs.

Local tracking has no MLflow dependency. To use MLflow, install it separately
and supply a tracking URI. An MLflow error is raised unless a local fallback
directory is explicitly configured; a fallback can leave a partial MLflow run.
The returned record identifies the actual backend and any partial run.

Supply only patient-independent metadata and aggregate metrics. Prompt
metadata is optional and must include a patient-independent template and its
matching SHA-256; the utility does not redact supplied values. Do not include
patient data or secrets. Local manifests have private file permissions.

.. autofunction:: pyhealth.experiment_tracking.record_experiment
