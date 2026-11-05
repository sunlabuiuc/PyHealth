import contextlib
import functools
import hashlib
import json
import os
import pickle
import random
import shutil
import subprocess
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
    """Seeds Python's ``random``, NumPy and PyTorch (CPU and CUDA).

    This is the only place PyHealth seeds the global random generators, and
    only when you call it. ``torch.manual_seed(seed)`` alone is also enough
    for reproducible PyHealth training on CPU: shuffling loaders draw their
    seed from torch's generator (see :func:`~pyhealth.datasets.get_dataloader`).

    On CUDA this also sets ``torch.backends.cudnn.deterministic = True`` and
    ``torch.backends.cudnn.benchmark = False``, which apply to the whole
    process. For bit-identical GPU runs you may additionally need
    ``torch.use_deterministic_algorithms(True)`` and the environment variable
    ``CUBLAS_WORKSPACE_CONFIG=:4096:8``.

    ``PYTHONHASHSEED`` is set in ``os.environ``; it does not change string
    hashing in the running interpreter, only in subprocesses started
    afterwards (such as ``set_task`` workers).

    Args:
        seed: The seed.

    Examples:
        >>> import random
        >>> from pyhealth.utils import set_seed
        >>> set_seed(0); a = random.random()
        >>> set_seed(0); b = random.random()
        >>> a == b
        True
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


@contextlib.contextmanager
def preserve_rng_state():
    """Restores the global random states after the block, whatever it did.

    Saves and restores Python's ``random``, NumPy's legacy global generator,
    PyTorch's CPU generator and, if CUDA is initialized, every CUDA
    generator. PyHealth wraps cache writes in it because litdata reseeds
    the global generators while writing, which would otherwise silently
    replace the caller's seed.

    Examples:
        >>> import random
        >>> from pyhealth.utils import preserve_rng_state
        >>> random.seed(1); state = random.getstate()
        >>> with preserve_rng_state():
        ...     random.seed(42)
        >>> random.getstate() == state
        True
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() and torch.cuda.is_initialized()
        else None
    )
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _seeded_by(attribute: str):
    """Method decorator: seed torch from ``self.<attribute>`` for the call only.

    If the attribute is None the method runs unchanged; otherwise torch is
    seeded inside :func:`preserve_rng_state`, so the call is reproducible and
    the caller's global random state is left as it was.
    """

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            seed = getattr(self, attribute, None)
            if seed is None:
                return method(self, *args, **kwargs)
            with preserve_rng_state():
                torch.manual_seed(seed)
                return method(self, *args, **kwargs)

        return wrapper

    return decorator


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_pickle(filename):
    """Load pickled data from a file.

    Example:
        >>> import tempfile
        >>> path = os.path.join(tempfile.mkdtemp(), "x.pkl")
        >>> _ = open(path, "wb").write(pickle.dumps({"a": 1}))
        >>> load_pickle(path)
        {'a': 1}
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    """Write data as pickle to a file.

    Example:
        >>> import tempfile
        >>> path = os.path.join(tempfile.mkdtemp(), "x.pkl")
        >>> save_pickle({"a": 1}, path) is None
        True
        >>> load_pickle(path)
        {'a': 1}
    """
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_json(filename):
    """Load JSON data from a file.

    Example:
        >>> import tempfile
        >>> path = os.path.join(tempfile.mkdtemp(), "x.json")
        >>> _ = open(path, "w").write('{"a": 1}')
        >>> load_json(path)
        {'a': 1}
    """
    with open(filename, "r") as f:
        return json.load(f)


def save_json(data, filename):
    """Write data as JSON to a file.

    Example:
        >>> import tempfile
        >>> path = os.path.join(tempfile.mkdtemp(), "x.json")
        >>> save_json({"a": 1}, path) is None
        True
        >>> load_json(path)
        {'a': 1}
    """
    with open(filename, "w") as f:
        json.dump(data, f)


def _git_revision():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, stderr=subprocess.DEVNULL
        ).decode().strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:  # noqa: BLE001 - provenance must never break a run
        return {"commit": None, "dirty": None}


def _source_digest():
    """Hash the package's ``.py`` sources so code identity survives deploy.

    Only ``.py`` files under the package are hashed: YAML configs, SQL,
    shell launchers, and tokenizer/model artifacts are not covered (an
    edited config in a no-git tarball is undetectable beyond its recorded
    path). Cluster runs typically execute from an unpacked tarball rather
    than a clone, so the git lookup returns nothing exactly where
    provenance matters most. Hashing the sources keeps "which code
    produced this result" answerable either way.
    """
    package = os.path.dirname(os.path.abspath(__file__))
    digest = hashlib.sha256()
    try:
        for root, dirs, files in os.walk(package):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for name in sorted(files):
                if not name.endswith(".py"):
                    continue
                path = os.path.join(root, name)
                digest.update(os.path.relpath(path, package).encode())
                with open(path, "rb") as f:
                    digest.update(f.read())
        return digest.hexdigest()
    except OSError:
        return None


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def write_run_config(exp_path, config):
    """Persist the resolved run configuration next to the run's metrics.

    ``metrics_history.json`` records what a run scored but not the conditions
    that produced it. Record the resolved settings, not the raw flags, so
    derived conditions (lr, split mode, eval split) are recoverable.

    Examples:
        >>> import json, tempfile
        >>> path = write_run_config(tempfile.mkdtemp(), {"lr": 1e-4})
        >>> sorted(json.load(open(path))["config"])
        ['lr']
    """
    record = {
        "config": {str(k): _jsonable(v) for k, v in config.items()},
        "git": _git_revision(),
        "source_sha256": _source_digest(),
        "torch": torch.__version__,
    }
    os.makedirs(exp_path, exist_ok=True)
    path = os.path.join(exp_path, "run_config.json")
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            json.dump(record, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


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