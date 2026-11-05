"""Subject-id normalization shared by the EEG tasks.

Author: Jathurshan Pradeepkumar (jp65@illinois.edu), UIUC
"""


def normalize_prefixed_subject(
    subject: str, prefix: str, width: int, separator: str = ""
) -> str:
    """Canonicalize a subject id of the form ``<prefix><separator><number>``.

    ``"1"``, ``"7"`` and ``"<prefix>7"`` all gain the prefix and zero
    padding. A value that is not numeric after the prefix (for example a
    custom cohort name) is lower-cased and returned unchanged.

    Args:
        subject: Subject id as stored on a sample or passed by the user.
        prefix: Lower-case id prefix, for example ``"chb"`` or ``"sub"``.
        width: Zero-padded width of the number.
        separator: Text between prefix and number in the canonical id,
            for example ``"-"`` for ``sub-001``. Also stripped from the
            input when present.

    Returns:
        The canonical subject id.

    Examples:
        >>> from pyhealth.tasks.subject_ids import normalize_prefixed_subject
        >>> normalize_prefixed_subject("7", "sub", 3, "-")
        'sub-007'
        >>> normalize_prefixed_subject("CHB2", "chb", 2)
        'chb02'
    """
    text = str(subject).strip().lower()
    if text.startswith(prefix):
        rest = text[len(prefix) :]
        if separator:
            rest = rest.lstrip(separator)
        if rest.isdigit():
            return f"{prefix}{separator}{int(rest):0{width}d}"
        return text
    if text.isdigit():
        return f"{prefix}{separator}{int(text):0{width}d}"
    return text
