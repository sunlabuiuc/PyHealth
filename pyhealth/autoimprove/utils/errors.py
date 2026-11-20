"""Errors as short text, with and without their message."""

from __future__ import annotations

import os
import traceback


def describe_error(error: BaseException, message: bool = True) -> str:
    """An error as text, with the line that raised it.

    A bare ``assert`` has no message, so the failing line is the only clue a
    proposer gets about what to change. With ``message=False`` the message is
    left out: an exception raised while reading a sample can quote a raw
    value, so this is the form shown to a proposer.

    Examples:
        >>> try:
        ...     assert 197 % 4 == 0
        ... except AssertionError as e:
        ...     print(describe_error(e))  # doctest: +ELLIPSIS
        AssertionError: (no message) at ...: assert 197 % 4 == 0
        >>> def parse(value):
        ...     return int(value)
        >>> try:
        ...     parse("a raw value")
        ... except ValueError as e:
        ...     print(describe_error(e, message=False))  # doctest: +ELLIPSIS
        ValueError at ...: return int(value)
    """
    text = type(error).__name__
    if message:
        text += f": {str(error)[:300] or '(no message)'}"
    frames = traceback.extract_tb(error.__traceback__)
    if frames:
        last = frames[-1]
        text += f" at {os.path.basename(last.filename)}:{last.lineno}"
        if last.line:
            text += f": {last.line[:200]}"
    return text
