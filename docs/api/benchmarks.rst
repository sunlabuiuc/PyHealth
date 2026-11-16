Benchmarks
==========

:mod:`pyhealth.benchmarks` provides the base abstractions for benchmarks. A
benchmark defines and executes evaluations. It knows nothing about runners,
storage or leaderboards.

An :class:`~pyhealth.benchmarks.Evaluation` is one unit of work, identified by
a hash of its parameters. A :class:`~pyhealth.benchmarks.Benchmark` runs a
batch of evaluations and returns one :class:`~pyhealth.benchmarks.Result` per
evaluation.

.. autoclass:: pyhealth.benchmarks.Evaluation
    :members:

.. autoclass:: pyhealth.benchmarks.Result
    :members:

.. autoclass:: pyhealth.benchmarks.Benchmark
    :members:
    :private-members: _evaluate
