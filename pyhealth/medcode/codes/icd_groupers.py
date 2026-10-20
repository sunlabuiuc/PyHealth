"""ICD grouper vocabularies backed by the ``icd-mappings`` package.

These are flat label systems rather than ontologies, so they subclass
:class:`~pyhealth.medcode.FlatMap`. Each class normalizes the upstream
representation into a canonical PyHealth code via ``standardize()``.

Author: John Wu (johnwu3)
"""

from typing import Any

from pyhealth.medcode.flat_map import FlatMap


def _strip_label(code: Any) -> str:
    """Reduces an ``"I00-I99 | Description"`` value to its code range."""
    text = str(code)
    return text.split("|")[0].strip()


def _indicator_to_code(code: Any) -> str:
    """Reduces a boolean chronic-condition indicator to ``"1"`` / ``"0"``."""
    if isinstance(code, bool):
        return "1" if code else "0"
    return str(code).strip()


class CCSR(FlatMap):
    """Clinical Classifications Software Refined, for ICD-10-CM.

    AHRQ's successor to CCS: 530 categories, ICD-10-CM only. Codes look like
    ``"CIR019"`` -- a body-system prefix plus an index.

    Examples:
        >>> from pyhealth.medcode import CrossMap
        >>> ccsr = CrossMap.load("ICD10CM", "CCSR")
        >>> ccsr.map("I50.9")          # heart failure
        ['CIR019']
        >>> ccsr.map("E11.9")          # type 2 diabetes
        ['END002']
        >>> ccsr.map("J18.9")          # pneumonia
        ['RSP002']
        >>> ccsr.map("N17.9")          # acute kidney failure
        ['GEN002']
    """

    def __init__(self, **kwargs):
        super().__init__(vocabulary="CCSR", **kwargs)


class CCI(FlatMap):
    """Chronic Condition Indicator, for ICD-9-CM.

    A binary flag: is this diagnosis a chronic condition? The upstream package
    returns a Python ``bool``; PyHealth normalizes it to ``"1"`` or ``"0"`` so
    that the vocabulary behaves like every other code system.

    Examples:
        >>> from pyhealth.medcode import CrossMap
        >>> cci = CrossMap.load("ICD9CM", "CCI")
        >>> cci.map("428.0")           # heart failure is chronic
        ['1']
        >>> cci.map("250.00")          # so is diabetes
        ['1']
        >>> cci.map("486")             # pneumonia is not
        ['0']
    """

    def __init__(self, **kwargs):
        super().__init__(vocabulary="CCI", **kwargs)

    @staticmethod
    def standardize(code: Any) -> str:
        """Standardizes a CCI indicator to ``"1"`` or ``"0"``."""
        return _indicator_to_code(code)


class CCIR(FlatMap):
    """Chronic Condition Indicator Refined, for ICD-10-CM.

    The ICD-10 counterpart of :class:`CCI`, normalized the same way.

    Examples:
        >>> from pyhealth.medcode import CrossMap
        >>> ccir = CrossMap.load("ICD10CM", "CCIR")
        >>> ccir.map("I50.9")          # heart failure is chronic
        ['1']
        >>> ccir.map("S72.001A")       # a femur fracture is acute
        ['0']
    """

    def __init__(self, **kwargs):
        super().__init__(vocabulary="CCIR", **kwargs)

    @staticmethod
    def standardize(code: Any) -> str:
        """Standardizes a CCIR indicator to ``"1"`` or ``"0"``."""
        return _indicator_to_code(code)


class ICD9CHAPTER(FlatMap):
    """Chapters of ICD-9-CM: 19 top-level groupings, numbered ``"1"``-``"19"``.

    Kept separate from :class:`ICD10CHAPTER` because the two chapter systems
    are different codespaces despite the shared name.

    Examples:
        >>> from pyhealth.medcode import CrossMap
        >>> chapters = CrossMap.load("ICD9CM", "ICD9CHAPTER")
        >>> chapters.map("428.0")      # 7 = circulatory system
        ['7']
        >>> chapters.map("038.9")      # 1 = infectious and parasitic
        ['1']
        >>> chapters.map("800.0")      # 17 = injury and poisoning
        ['17']
    """

    def __init__(self, **kwargs):
        super().__init__(vocabulary="ICD9CHAPTER", **kwargs)


class ICD10CHAPTER(FlatMap):
    """Chapters of ICD-10-CM: 22 top-level groupings, keyed by code range.

    The upstream package returns ``"I00-I99 | Diseases of the circulatory
    system"``. PyHealth keeps the range (``"I00-I99"``) as the code, since the
    description is a label rather than an identifier.

    Examples:
        >>> from pyhealth.medcode import CrossMap
        >>> chapters = CrossMap.load("ICD10CM", "ICD10CHAPTER")
        >>> chapters.map("I50.9")      # circulatory system
        ['I00-I99']
        >>> chapters.map("E11.9")      # endocrine, nutritional, metabolic
        ['E00-E89']
        >>> chapters.map("A41.9")      # infectious and parasitic
        ['A00-B99']
    """

    def __init__(self, **kwargs):
        super().__init__(vocabulary="ICD10CHAPTER", **kwargs)

    @staticmethod
    def standardize(code: Any) -> str:
        """Reduces an ICD-10 chapter value to its code range."""
        return _strip_label(code)


class ICD10BLOCK(FlatMap):
    """Blocks of ICD-10-CM: 226 groupings nested inside chapters.

    Normalized the same way as :class:`ICD10CHAPTER`.

    Examples:
        >>> from pyhealth.medcode import CrossMap
        >>> blocks = CrossMap.load("ICD10CM", "ICD10BLOCK")
        >>> blocks.map("I50.9")        # other forms of heart disease
        ['I30-I5A']
        >>> blocks.map("E11.9")        # diabetes mellitus
        ['E08-E13']
        >>> blocks.map("A41.9")        # other bacterial diseases
        ['A30-A49']
    """

    def __init__(self, **kwargs):
        super().__init__(vocabulary="ICD10BLOCK", **kwargs)

    @staticmethod
    def standardize(code: Any) -> str:
        """Reduces an ICD-10 block value to its code range."""
        return _strip_label(code)


class CCC(FlatMap):
    """Pediatric Complex Chronic Condition category.

    Sparse by design: most diagnoses are not complex chronic conditions and
    map to no category at all, which surfaces as an empty list.

    Examples:
        >>> from pyhealth.medcode import CrossMap
        >>> ccc = CrossMap.load("ICD10CM", "CCC")
        >>> ccc.map("I50.9")           # cardiovascular
        ['cvd']
        >>> ccc.map("J18.9")           # pneumonia: not a complex chronic
        []
    """

    def __init__(self, **kwargs):
        super().__init__(vocabulary="CCC", **kwargs)

    @staticmethod
    def standardize(code: Any) -> str:
        """Standardizes a CCC category by trimming surrounding whitespace."""
        return str(code).strip()


class CCCSUB(FlatMap):
    """Pediatric Complex Chronic Condition subcategory.

    The finer level beneath :class:`CCC`. Upstream values carry trailing
    whitespace, which is trimmed here.

    Examples:
        >>> from pyhealth.medcode import CrossMap
        >>> sub_map = CrossMap.load("ICD10CM", "CCCSUB")
        >>> sub_map.map("I50.9")
        ['Other Cardiovascular']
        >>> sub_map.map("J18.9")
        []
    """

    def __init__(self, **kwargs):
        super().__init__(vocabulary="CCCSUB", **kwargs)

    @staticmethod
    def standardize(code: Any) -> str:
        """Standardizes a CCC subcategory by trimming surrounding whitespace."""
        return str(code).strip()
