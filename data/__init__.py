from .data import Event, Patient


class Visit:
    """This class is deprecated and should not be used."""
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn("The Visit class is deprecated and will be removed in a future version.", DeprecationWarning)