pyhealth.tasks.LNDbTextureClassification
========================================

Texture characterisation of LNDb nodules >= 3 mm, the LNDb challenge's
nodule texture task: ground-glass opacity (GGO), part-solid or solid, from
the mean radiologist rating with the official thresholds (< 7/3 GGO, 7/3 to
11/3 part-solid, > 11/3 solid). Patches are 3D cubes or 2.5D orthogonal
planes, as in :class:`~pyhealth.tasks.LNDbFindingClassification`.

.. autoclass:: pyhealth.tasks.LNDbTextureClassification
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.lndb_texture.lndb_texture_class
