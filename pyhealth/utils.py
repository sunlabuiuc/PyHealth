import hashlib
import json
import os
import pickle
import random
import subprocess
import contextlib

import numpy as np
import torch


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
    except Exception:
        return {"commit": None, "dirty": None}


def _source_digest():
    """Hash the package source so code identity survives a non-git deploy.

    Cluster runs typically execute from an unpacked tarball rather than a
    clone, so the git lookup returns nothing exactly where provenance matters
    most. Hashing the sources keeps "which code produced this result"
    answerable either way.
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