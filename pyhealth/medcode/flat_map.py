"""Flat (non-hierarchical) medical code systems.

Author: John Wu (johnwu3)
"""

from abc import ABC, abstractmethod


class FlatMap(ABC):
    """Contains information for a flat, non-hierarchical code system.

    Some code systems are *groupers*: they assign a code to a category drawn
    from a small, flat label set. CCSR, the AHRQ chronic condition indicators,
    and the ICD chapter/block systems are all of this kind. They have no
    ``code``/``parent_code`` ontology file on PyHealth's resource server, so
    they cannot be :class:`~pyhealth.medcode.InnerMap` subclasses -- there is
    no graph to build.

    ``FlatMap`` supplies exactly the surface that
    :class:`~pyhealth.medcode.CrossMap` requires of a vocabulary --
    ``standardize()`` and ``convert()`` -- and nothing more. It never touches
    the network.

    Because the graph-backed methods of ``InnerMap`` (``lookup``,
    ``get_ancestors``, ``get_descendants``, ``stat``) have no meaning without
    an ontology, they are deliberately absent and raise ``AttributeError``.

    Examples:
        >>> from pyhealth.medcode import FlatMap
        >>> ccsr = FlatMap.load("CCSR")
        >>> ccsr.vocabulary
        'CCSR'
        >>> ccsr.standardize("CIR019")
        'CIR019'
    """

    @abstractmethod
    def __init__(self, vocabulary: str, refresh_cache: bool = False):
        """Initializes the flat code system.

        Args:
            vocabulary: the name of the code system, e.g. ``"CCSR"``.
            refresh_cache: accepted and ignored. There is nothing to cache;
                the argument exists so that ``InnerMap.load()``, which always
                passes it, works for these vocabularies too.
        """
        self.vocabulary = vocabulary

    @classmethod
    def load(cls, vocabulary: str, refresh_cache: bool = False) -> "FlatMap":
        """Initializes a flat code system by name.

        Args:
            vocabulary: the name of the code system, e.g. ``"CCSR"``.
            refresh_cache: accepted and ignored.

        Returns:
            The vocabulary instance.

        Examples:
            >>> from pyhealth.medcode import FlatMap
            >>> chapters = FlatMap.load("ICD10CHAPTER")
            >>> chapters.vocabulary
            'ICD10CHAPTER'
        """
        from pyhealth import medcode

        target = getattr(medcode, vocabulary)
        return target(refresh_cache=refresh_cache)

    @staticmethod
    def standardize(code: str) -> str:
        """Standardizes a code. Grouper codes are already canonical."""
        return code

    @staticmethod
    def convert(code: str, **kwargs) -> str:
        """Converts a code. Grouper codes have no alternate representations."""
        return code

    def __repr__(self) -> str:
        return f"FlatMap(vocabulary={self.vocabulary})"
