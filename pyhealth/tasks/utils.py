from datetime import datetime  # Make sure we're using the class, not the module


def parse_datetime(datetime_str):
    """Parse a datetime string in various formats.

    Args:
        datetime_str: String representation of a datetime

    Returns:
        A datetime object or None if parsing fails
    """

    if datetime_str is None:
        return None

    # If it's already a datetime object, return it
    if isinstance(datetime_str, datetime):
        return datetime_str

    # Common format patterns to try
    formats = [
        "%Y-%m-%d %H:%M:%S",  # 2020-01-01 12:30:45
        "%Y-%m-%d",  # 2020-01-01
        "%d/%m/%Y %H:%M:%S",  # 01/01/2020 12:30:45
        "%d/%m/%Y",  # 01/01/2020
        "%Y%m%d%H%M%S",  # 20200101123045
        "%Y%m%d",  # 20200101
    ]

    # Try each format
    for fmt in formats:
        try:
            return datetime.strptime(str(datetime_str), fmt)
        except (ValueError, TypeError):
            continue

    # Log warning if all formats fail
    print(f"Warning: Could not parse datetime '{datetime_str}'")
    return None
