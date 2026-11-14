from .data import Event, Patient


class Visit:
    """Removed in PyHealth 2.0; read visits with ``Patient.get_events``.

    Creating an instance raises at once with the replacement, instead of
    returning an object with none of its attributes set.

    Example:
        >>> from pyhealth.data import Visit
        >>> try:
        ...     Visit()
        ... except NotImplementedError as err:
        ...     print(str(err).split(";")[0])
        Visit was removed in PyHealth 2.0
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Visit was removed in PyHealth 2.0; use "
            "Patient.get_events(event_type=...) to read a patient's events. "
            "Event types are the dataset's table names, which differ by "
            "dataset."
        )
