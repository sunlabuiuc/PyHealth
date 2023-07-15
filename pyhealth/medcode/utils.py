import logging
import os
from urllib.parse import urljoin
from urllib.request import urlretrieve

import pandas as pd

from pyhealth import BASE_CACHE_PATH
from pyhealth.utils import create_directory, load_pickle, load_json

BASE_URL = "https://storage.googleapis.com/pyhealth/resource/"
MODULE_CACHE_PATH = os.path.join(BASE_CACHE_PATH, "medcode")
create_directory(MODULE_CACHE_PATH)

logger = logging.getLogger(__name__)


def download_and_read_csv(filename: str, refresh_cache: bool = False) -> pd.DataFrame:
    """Reads a csv file from the pyhealth resource folder.

    This function will read the csv file from `MODULE_CACHE_PATH` if it exists.
    Otherwise, it will download the csv file from `BASE_URL` and save it to
    `MODULE_CACHE_PATH`.

    Args:
        filename: The name of the csv file.
        refresh_cache: Whether to refresh the cache. Default is False.

    Returns:
        A pandas DataFrame.
    """
    local_filepath = os.path.join(MODULE_CACHE_PATH, filename)
    online_filepath = urljoin(BASE_URL, filename)
    if (not os.path.exists(local_filepath)) or refresh_cache:
        logger.debug(f"downloading {online_filepath} to {local_filepath}")
        urlretrieve(online_filepath, local_filepath)
    return pd.read_csv(local_filepath, dtype=str)


embedding_types = ['KG/transe', 'LM/clinicalbert', 'LM/gpt3', 'LM/biogpt', 'LM/sapbert']
features = ['conditions', 'procedures', 'drugs']
for embedding_type in embedding_types:
    for feature in features:
        MODULE_CACHE_PATH_TMP = os.path.join(BASE_CACHE_PATH, "medcode", "embeddings", embedding_type, feature)
        create_directory(MODULE_CACHE_PATH_TMP)

    MODULE_CACHE_PATH_TMP = os.path.join(BASE_CACHE_PATH, "medcode", "embeddings", embedding_type, "special_tokens")
    create_directory(MODULE_CACHE_PATH_TMP)


def download_and_read_pkl(filename: str, refresh_cache: bool = False):
    """Reads a pickle file from the pyhealth resource folder.

    This function will read the pickle file from `MODULE_CACHE_PATH` if it
    exists. Otherwise, it will download the pickle file from `BASE_URL` and
    save it to `MODULE_CACHE_PATH`.

    Args:
        filename: The name of the pickle file.
        refresh_cache: Whether to refresh the cache. Default is False.

    Returns:
        The object loaded from the pickle file.
    """
    local_filepath = os.path.join(MODULE_CACHE_PATH, filename)
    online_filepath = urljoin(BASE_URL, filename)
    if (not os.path.exists(local_filepath)) or refresh_cache:
        logger.debug(f"downloading {online_filepath} to {local_filepath}")
        urlretrieve(online_filepath, local_filepath)
    return load_pickle(local_filepath)


def download_and_read_json(filename: str, refresh_cache: bool = True):
    """Reads a json file from the pyhealth resource folder.

    This function will read the json file from `MODULE_CACHE_PATH` if it exists.
    Otherwise, it will download the json file from `BASE_URL` and save it to
    `MODULE_CACHE_PATH`.

    Args:
        filename: The name of the json file.
        refresh_cache: Whether to refresh the cache. Default is False.

    Returns:
        The object loaded from the json file.
    """
    local_filepath = os.path.join(MODULE_CACHE_PATH, filename)
    online_filepath = urljoin(BASE_URL, filename)
    if (not os.path.exists(local_filepath)) or refresh_cache:
        logger.debug(f"downloading {online_filepath} to {local_filepath}")
        urlretrieve(online_filepath, local_filepath)
    return load_json(local_filepath)
