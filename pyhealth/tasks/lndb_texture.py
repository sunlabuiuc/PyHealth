"""LNDb nodule texture classification (GGO / part-solid / solid)."""

from __future__ import annotations

from typing import Any, ClassVar

from .lndb_classification import LNDbFindingClassification

#: Class names of the texture label, by index.
LNDB_TEXTURE_CLASSES = ("ggo", "part_solid", "solid")

#: Thresholds on the mean texture rating (1-5) used by the official LNDb
#: ``calcNodTexClass`` script: < 7/3 GGO, 7/3 to 11/3 part-solid, > 11/3 solid.
LNDB_TEXTURE_THRESHOLDS = (7.0 / 3.0, 11.0 / 3.0)


def lndb_texture_class(rating: float) -> int:
    """Map a mean LNDb texture rating (1-5) to 0 GGO, 1 part-solid, 2 solid.

    Follows ``calcNodTexClass`` in the official LNDb ``scripts.zip``
    (``texthr = [7/3, 11/3]``, both bounds inclusive for part-solid).

    Args:
        rating: Mean texture rating of the radiologists, 1 (GGO) to 5 (solid).

    Returns:
        The texture class index of :data:`LNDB_TEXTURE_CLASSES`.

    Examples:
        >>> [lndb_texture_class(r) for r in (1.0, 2.0, 2.5, 3.0, 11 / 3, 4.0, 5.0)]
        [0, 0, 1, 1, 1, 2, 2]
    """
    low, high = LNDB_TEXTURE_THRESHOLDS
    if rating < low:
        return 0
    if rating <= high:
        return 1
    return 2


class LNDbTextureClassification(LNDbFindingClassification):
    """Classify the texture of an LNDb nodule >= 3 mm from a CT patch.

    Works with :class:`~pyhealth.datasets.LNDbDataset`.

    This is the nodule texture characterisation task of the LNDb challenge
    (Pedrosa et al., 2021): ground-glass opacity (GGO), part-solid or solid,
    the texture classes the Fleischner guidelines use. Only nodules >= 3 mm
    (``finding_class == 2``) become samples, as in the official
    ``getNoduleCubes.py`` script (nodules < 3 mm were marked by their centre
    only). The label ``texture_class`` comes from the mean radiologist
    rating with the official thresholds of :func:`lndb_texture_class`. It is
    heavily imbalanced (the 625 training nodules >= 3 mm: 33 GGO, 55
    part-solid, 537 solid), so report macro F1 or balanced accuracy.

    Patches are cut as in :class:`LNDbFindingClassification`
    (``patch_mode="3d"`` or ``"2.5d"``).

    Args:
        patch_size: Edge length of the patch in voxels. Defaults to 32.
        hu_window: ``(low, high)`` Hounsfield range mapped to [0, 1].
            Defaults to ``(-1000, 400)``.
        patch_mode: ``"3d"`` or ``"2.5d"``.
        min_agreement: Keep nodules marked by at least this many
            radiologists. Defaults to 1.

    Examples:
        >>> from pyhealth.tasks import LNDbTextureClassification
        >>> task = LNDbTextureClassification(patch_mode="2.5d")
        >>> task.output_schema
        {'texture_class': ('multiclass', {'labels': 3})}
        >>> from pyhealth.datasets import LNDbDataset
        >>> dataset = LNDbDataset(root="/data/lndb")  # doctest: +SKIP
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
    """

    task_name: str = "LNDbTextureClassification"
    output_schema: ClassVar[dict[str, Any]] = {
        "texture_class": ("multiclass", {"labels": len(LNDB_TEXTURE_CLASSES)})
    }

    def __init__(
        self,
        patch_size: int = 32,
        hu_window: tuple[float, float] = (-1000.0, 400.0),
        patch_mode: str = "3d",
        min_agreement: int = 1,
    ) -> None:
        super().__init__(
            patch_size=patch_size,
            hu_window=hu_window,
            patch_mode=patch_mode,
            binary=False,
            min_agreement=min_agreement,
        )
        self.label_key = "texture_class"
        self.output_schema = {
            "texture_class": (
                "multiclass",
                {"labels": len(LNDB_TEXTURE_CLASSES)},
            )
        }

    def keep(self, event: Any) -> bool:
        """Keep nodules >= 3 mm that have a texture rating.

        Examples:
            >>> from types import SimpleNamespace
            >>> task = LNDbTextureClassification()
            >>> task.keep(SimpleNamespace(finding_class="2", texture="4.5"))
            True
            >>> task.keep(SimpleNamespace(finding_class="0", texture=""))
            False
        """
        texture = getattr(event, "texture", None)
        if texture is None or str(texture).strip() in ("", "nan", "None"):
            return False
        return int(float(event.finding_class)) == 2

    def label(self, event: Any) -> int:
        """Return the texture class of a nodule event.

        Examples:
            >>> from types import SimpleNamespace
            >>> LNDbTextureClassification().label(SimpleNamespace(texture="3.0"))
            1
        """
        return lndb_texture_class(float(event.texture))
