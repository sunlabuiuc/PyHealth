pyhealth.tasks.fingerprint
=======================================

Deterministic cache keys for ``set_task``. Replaces the previous
``json.dumps(..., default=str)`` hash, which was neither stable across
processes nor injective for large arrays.

.. automodule:: pyhealth.tasks.fingerprint
    :members: UnfingerprintableError, task_spec, task_fingerprint, task_cache_name, processors_fingerprint, write_task_metadata, slugify, record_init_args
    :undoc-members:
    :show-inheritance:
