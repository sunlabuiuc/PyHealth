pyhealth.models.Synthea
===================================

Synthea: a rule-based synthetic patient simulator. Unlike the trained
generators (HALO, MedGAN, ...), Synthea is not fitted to data; it simulates
patient lifetimes from clinical care modules. This class runs the pinned
Synthea release as a Java subprocess and writes CSV files to disk. It subclasses
:class:`~pyhealth.models.BaseModel` for API consistency but has no trainable
parameters and no forward pass; load its output with
:class:`~pyhealth.datasets.SyntheaCSVDataset`.

Requirements
------------

**Java 17 or newer** must be installed separately; ``pip install pyhealth``
does not provide it. Synthea looks for Java in this order:

1. the ``java_path`` argument,
2. ``$JAVA_HOME/bin/java``,
3. ``java`` on ``PATH``.

The first one found is used. If none is found, or it is older than Java 17,
:meth:`~pyhealth.models.Synthea.run` raises a ``RuntimeError`` naming the
executable and its version. Any JDK 17+ distribution works, for example
`Eclipse Temurin <https://adoptium.net>`_, or ``conda install openjdk``.
Constructing the class and calling
:meth:`~pyhealth.models.Synthea.build_argv` do not need Java.

The Synthea JAR itself (about 200 MB) is downloaded and checksum-verified on
first use unless ``jar_path`` is given.

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
