import json
import os
import pickle
import random
import contextlib
import shutil
import urllib.request

import numpy as np
import torch


def download_file(url: str, path: str, timeout: float = 60.0) -> str:
    """Downloads ``url`` to ``path`` without leaving a partial file behind.

    The file is written to ``path + ".part"`` and moved into place only after
    the whole response has been read, so an interrupted download never
    produces a truncated file that later calls would mistake for a valid
    cache. ``timeout`` (seconds) applies to connecting and to each read.

    Args:
        url: The URL to download.
        path: Destination file path.
        timeout: Socket timeout in seconds. Default is 60.

    Returns:
        ``path``.

    Examples:
        >>> from pyhealth.utils import download_file
        >>> download_file(  # doctest: +SKIP
        ...     "https://storage.googleapis.com/pyhealth/resource/ICD9CM.csv",
        ...     "/tmp/ICD9CM.csv",
        ... )
        '/tmp/ICD9CM.csv'
    """
    path = str(path)
    part = path + ".part"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, open(
            part, "wb"
        ) as f:
            shutil.copyfileobj(response, f)
            expected = response.headers.get("Content-Length")
            received = f.tell()
        # A dropped connection ends the read early without raising.
        if expected is not None and received != int(expected):
            raise OSError(
                f"Incomplete download of {url}: got {received} of {expected} bytes"
            )
        os.replace(part, path)
    finally:
        if os.path.exists(part):
            os.remove(part)
    return path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)

@contextlib.contextmanager
def set_env(**environ):
    """
    Temporarily set the process environment variables.

    >>> with set_env(PLUGINS_DIR='test/plugins'):
    ...   "PLUGINS_DIR" in os.environ
    True

    >>> "PLUGINS_DIR" in os.environ
    False

    :type environ: dict[str, unicode]
    :param environ: Environment variables to set
    """
    old_environ = dict(os.environ)
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)