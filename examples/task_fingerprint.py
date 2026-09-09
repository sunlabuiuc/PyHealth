"""Show that task cache keys include init args (issue #916).

Two ReadmissionPredictionMIMIC3 configurations that differ only by
``window`` must produce different cache directory names. A slash in
``task_name`` (as used by BenchmarkEHRShot) is slugified to a single
path component.
"""

from datetime import timedelta

from pyhealth.tasks.benchmark_ehrshot import BenchmarkEHRShot
from pyhealth.tasks.fingerprint import task_cache_name, task_fingerprint
from pyhealth.tasks.readmission_prediction import ReadmissionPredictionMIMIC3

if __name__ == "__main__":
    t15 = ReadmissionPredictionMIMIC3()
    t30 = ReadmissionPredictionMIMIC3(window=timedelta(days=30))
    print("15-day window:", task_cache_name(t15))
    print("30-day window:", task_cache_name(t30))
    assert task_fingerprint(t15) != task_fingerprint(t30)

    ehrshot = task_cache_name(BenchmarkEHRShot(task="guo_los"))
    print("EHRShot guo_los:", ehrshot)
    assert "/" not in ehrshot
