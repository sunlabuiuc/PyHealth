pyhealth.models.Synthea
===================================

`Synthea <https://github.com/synthetichealth/synthea>`_ is a rule-based
synthetic patient simulator from MITRE. Unlike the trained generators (HALO,
MedGAN, ...), Synthea is not fitted to data; it simulates patient lifetimes from
clinical care modules. This class runs the pinned Synthea release as a Java
subprocess and writes CSV files to disk. It subclasses
:class:`~pyhealth.models.BaseModel` for API consistency but has no trainable
parameters and no forward pass; load its output with
:class:`~pyhealth.datasets.SyntheaCSVDataset`.

Quick Start
-----------

The constructor only sets up the generator: where output goes, and where to
find Java and the Synthea JAR. Nothing runs until
:meth:`~pyhealth.models.Synthea.generate`, which takes the population settings
and returns the directory of CSV files that
:class:`~pyhealth.datasets.SyntheaCSVDataset` takes as ``root``.

.. code-block:: python

    from pyhealth.datasets import SyntheaCSVDataset
    from pyhealth.models import Synthea

    synthea = Synthea("./synthea-output")  # setup only; nothing runs yet

    csv_dir = synthea.generate(population=100, seed=42)  # runs Java
    dataset = SyntheaCSVDataset(csv_dir, tables=["conditions", "medications"])
    dataset.stats()

    # The same instance can generate as many populations as you like.
    larger = synthea.generate(population=1000, seed=42)
    ohio = synthea.generate(population=100, seed=42, state="Ohio")

Each combination of settings is written to its own subdirectory of
``output_dir``, so populations never overwrite each other::

    synthea-output/
    └── 3f9c...e1/            # SHA-256 of the generation settings
        └── csv/
            ├── patients.csv
            ├── conditions.csv
            └── ...

Calling ``generate()`` again with the same settings returns the existing
directory without rerunning Synthea, which can take minutes. Pass
``overwrite=True`` to force a fresh run. If you omit ``seed``, a random seed is
chosen and logged, so every call produces a new population; pass a seed when
you need reproducible output.

See :doc:`../datasets/pyhealth.datasets.SyntheaCSVDataset` for what the CSV
files contain and how they differ from other PyHealth datasets.

Setup
-----

You do **not** need to download or build Synthea yourself. The only manual
step is installing Java.

**1. Install Java 17 or newer.** ``pip install pyhealth`` does not provide it.
Any JDK distribution works, for example:

- conda: ``conda install -c conda-forge openjdk``
- macOS (Homebrew): ``brew install openjdk``
- Ubuntu/Debian: ``sudo apt install openjdk-17-jre-headless``
- Windows or any platform: an installer from
  `Eclipse Temurin <https://adoptium.net>`_

Check the installation with ``java -version``; the first line should report
version 17 or higher.

**2. Generate.** On the first call to :meth:`~pyhealth.models.Synthea.generate`,
PyHealth downloads the pinned Synthea release JAR (v4.0.0, about 200 MB),
verifies its SHA-256 checksum, and caches it under the PyHealth cache
directory:

- Linux: ``~/.cache/pyhealth/synthea/``
- macOS: ``~/Library/Caches/pyhealth/synthea/``
- Windows: ``%LOCALAPPDATA%\pyhealth\pyhealth\Cache\synthea\``

Later runs reuse the cached JAR.

**Offline machines or a custom Synthea build.** Download
``synthea-with-dependencies.jar`` from the
`Synthea releases page <https://github.com/synthetichealth/synthea/releases>`_
on a machine with internet access, copy it over, and pass its path:

.. code-block:: python

    synthea = Synthea(
        "./synthea-output",
        jar_path="/path/to/synthea-with-dependencies.jar",
        auto_download=False,
    )

**How Java is found.** The first of these that exists is used:

1. the ``java_path`` argument,
2. ``$JAVA_HOME/bin/java``,
3. ``java`` on ``PATH``.

If none is found, or it is older than Java 17,
:meth:`~pyhealth.models.Synthea.generate` raises a ``RuntimeError`` naming the
executable and its version. Constructing the class and calling
:meth:`~pyhealth.models.Synthea.build_argv` do not need Java.

Choosing What to Generate
-------------------------

Synthea's command-line options are arguments of
:meth:`~pyhealth.models.Synthea.generate` (``population``, ``seed``,
``state``, ``city``, ``gender``, ``age_range``, ``reference_date``, ...).
Anything from Synthea's
`synthea.properties <https://github.com/synthetichealth/synthea/wiki/Common-Configuration>`_
file goes in ``synthea_config``. Use
:meth:`~pyhealth.models.Synthea.get_available_config` to list the properties
the JAR accepts.

.. code-block:: python

    csv_dir = synthea.generate(
        population=500,
        seed=7,
        state="Massachusetts",
        age_range="40-80",
        synthea_config={
            "generate.only_alive_patients": True,
            "exporter.years_of_history": 10,
        },
    )

CSV export is switched on automatically and other export formats (FHIR, C-CDA,
...) are switched off. ``synthea_config`` cannot change which exporters run or
where output is written; use ``output_dir`` for that.

To check the exact command without running anything (no Java needed), call
:meth:`~pyhealth.models.Synthea.build_argv` with the same settings::

    synthea.build_argv("java", "synthea.jar", population=500, seed=7)

Reference:
    Walonoski, J., Kramer, M., Nichols, J., Quina, A., Moesel, C., Hall, D.,
    Duffett, C., Dube, K., Gallagher, T., & McLachlan, S. (2018).
    *Synthea: An approach, method, and software mechanism for generating
    synthetic patients and the synthetic electronic health care record.*
    Journal of the American Medical Informatics Association, 25(3), 230-238.
    https://doi.org/10.1093/jamia/ocx079

.. autoclass:: pyhealth.models.Synthea
    :members:
    :undoc-members:
    :show-inheritance:
