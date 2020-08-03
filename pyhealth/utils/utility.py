# -*- coding: utf-8 -*-
"""A set of utility functions to support outlier detection.
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
from numpy import percentile
import numbers

import sklearn
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler

from sklearn.utils import column_or_1d
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length

from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement

MAX_INT = np.iinfo(np.int32).max
MIN_INT = -1 * MAX_INT

def make_dirs_if_not_exists(save_dir):
    # make saving directory if needed
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

def read_csv_to_df(file_loc, header_lower=True, usecols=None, dtype=None,
                   low_memory=True, encoding=None):
    """Read in csv files with necessary processing

    Parameters
    ----------
    file_loc
    header_lower
    low_memory

    Returns
    -------

    """
    if dtype != None:
        df = pd.read_csv(file_loc, usecols=usecols, dtype=dtype,
                         low_memory=low_memory, encoding=encoding)
    else:
        df = pd.read_csv(file_loc, usecols=usecols, low_memory=low_memory,
                         encoding=encoding)

    if header_lower:
        df.columns = df.columns.str.lower()
    return df


def read_excel_to_df(file_loc, header_lower=True, usecols=None, dtype=None,
                     low_memory=True, encoding=None):
    """Read in excel files with necessary processing

    Parameters
    ----------
    file_loc
    header_lower
    low_memory

    Returns
    -------

    """
    if dtype != None:
        df = pd.read_excel(file_loc, usecols=usecols, dtype=dtype,
                           low_memory=low_memory, encoding=encoding)
    else:
        df = pd.read_excel(file_loc, usecols=usecols, low_memory=low_memory,
                           encoding=encoding)

    if header_lower:
        df.columns = df.columns.str.lower()
    return df


def check_parameter(param, low=MIN_INT, high=MAX_INT, param_name='',
                    include_left=False, include_right=False):
    """Check if an input is within the defined range.

    Parameters
    ----------
    param : int, float
        The input parameter to check.

    low : int, float
        The lower bound of the range.

    high : int, float
        The higher bound of the range.

    param_name : str, optional (default='')
        The name of the parameter.

    include_left : bool, optional (default=False)
        Whether includes the lower bound (lower bound <=).

    include_right : bool, optional (default=False)
        Whether includes the higher bound (<= higher bound).

    Returns
    -------
    within_range : bool or raise errors
        Whether the parameter is within the range of (low, high)

    """

    # param, low and high should all be numerical
    if not isinstance(param, (numbers.Integral, np.integer, np.float)):
        raise TypeError('{param_name} is set to {param} Not numerical'.format(
            param=param, param_name=param_name))

    if not isinstance(low, (numbers.Integral, np.integer, np.float)):
        raise TypeError('low is set to {low}. Not numerical'.format(low=low))

    if not isinstance(high, (numbers.Integral, np.integer, np.float)):
        raise TypeError('high is set to {high}. Not numerical'.format(
            high=high))

    # at least one of the bounds should be specified
    if low is MIN_INT and high is MAX_INT:
        raise ValueError('Neither low nor high bounds is undefined')

    # if wrong bound values are used
    if low > high:
        raise ValueError(
            'Lower bound > Higher bound')

    # value check under different bound conditions
    if (include_left and include_right) and (param < low or param > high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of [{low}, {high}].'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (include_left and not include_right) and (
            param < low or param >= high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of [{low}, {high}).'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (not include_left and include_right) and (
            param <= low or param > high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of ({low}, {high}].'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (not include_left and not include_right) and (
            param <= low or param >= high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of ({low}, {high}).'.format(
                param=param, low=low, high=high, param_name=param_name))
    else:
        return True
