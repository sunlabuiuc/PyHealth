from .data import Event, Patient


class Visit:
    """This class is deprecated and should not be used.

    Examples:
        >>> import warnings
        >>> from pyhealth.data import Visit
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     _ = Visit()
        >>> caught[0].category.__name__
        'DeprecationWarning'
    """
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn("The Visit class is deprecated and will be removed in a future version.", DeprecationWarning, stacklevel=2)