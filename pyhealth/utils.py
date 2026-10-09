import json
import os
import pickle
import random
import contextlib
import functools

import numpy as np
import torch


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