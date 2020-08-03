# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause
"""MIMIC dataset handling. Adapted and modified from
https://github.com/YerevaNN/mimic3-benchmarks/blob/master/mimic3benchmark/preprocessing.py
"""
import numpy as np
import pandas as pd
import re


# Temperature: map Farenheit to Celsius, some ambiguous 50<x<80
def clean_temperature(df):
    try:
        v = df.value.astype('float32').copy()
    except ValueError:
        print("could not convert string to float in clean_temperature. "
              "Set to NaN")
        v = df.value
        v[:] = np.nan
        return v
    else:
        idx = df.valueuom.fillna('').apply(lambda s: 'F' in s.lower()) | (
                v >= 79)
        v.loc[idx] = (v[idx] - 32) * 5. / 9
        return v


def clean_sbp(df):
    v = df.value.astype(str).copy()
    idx = v.apply(lambda s: '/' in s)
    v.loc[idx] = v[idx].apply(lambda s: re.match('^(\d+)/(\d+)$', s).group(1))
    return v.astype(float)


def clean_dbp(df):
    v = df.value.astype(str).copy()
    idx = v.apply(lambda s: '/' in s)
    v.loc[idx] = v[idx].apply(lambda s: re.match('^(\d+)/(\d+)$', s).group(2))
    return v.astype(float)


def clean_crr(df):
    v = pd.Series(np.zeros(df.shape[0]), index=df.index)
    v[:] = np.nan

    # when df.value is empty, dtype can be float and comparision with string
    # raises an exception, to fix this we change dtype to str
    df_value_str = df.value.astype(str)

    v.loc[(df_value_str == 'Normal <3 secs') | (df_value_str == 'Brisk')] = 0
    v.loc[
        (df_value_str == 'Abnormal >3 secs') | (df_value_str == 'Delayed')] = 1
    return v


# Weight: some really light/heavy adults: <50 lb, >450 lb, ambiguous oz/lb
# Children are tough for height, weight
def clean_weight(df):
    try:
        v = df.value.astype('float32').copy()
    except ValueError:
        print("could not convert string to float in clean_weight. Set to NaN")
        v = df.value
        v[:] = np.nan
        return v
    else:
        # ounces
        idx = df.valueuom.fillna('').apply(lambda s: 'oz' in s.lower())
        v.loc[idx] = v[idx] / 16.
        # pounds
        idx = idx | df.valueuom.fillna('').apply(lambda s: 'lb' in s.lower())
        v.loc[idx] = v[idx] * 0.453592
        return v


# Height: some really short/tall adults: <2 ft, >7 ft)
# Children are tough for height, weight
def clean_height(df):
    try:
        v = df.value.astype('float32').copy()
    except ValueError:
        print("could not convert string to float in clean_height. Set to NaN")
        v = df.value
        v[:] = np.nan
        return v
    else:
        idx = df.valueuom.fillna('').apply(lambda s: 'in' in s.lower())
        v.loc[idx] = np.round(v[idx] * 2.54)
        return v


# FIO2: many 0s, some 0<x<0.2 or 1<x<20
def clean_fio2(df):
    try:
        v = df.value.astype('float32').copy()
    except ValueError:
        print("could not convert string to float in clean_fio2. Set to NaN")
        v = df.value
        v[:] = np.nan
        return v
    else:
        ''' The line below is the correct way of doing the cleaning, 
        since we will not compare 'str' to 'float'.
        If we use that line it will create mismatches from the data of the 
        paper in ~50 ICU stays. The next releases of the benchmark should use this line.
        '''
        # idx = df.valueuom.fillna('').apply(lambda s: 'torr' not in s.lower()) & (v>1.0)

        ''' The line below was used to create the benchmark dataset that the 
        paper used. Note this line will not work in python 3, 
        since it may try to compare 'str' to 'float'.
        '''
        # idx = df.valueuom.fillna('').apply(lambda s: 'torr' not in s.lower()) & (df.value > 1.0)

        ''' The two following lines implement the code that was used to create 
        the benchmark dataset that the paper used.
        This works with both python 2 and python 3.
        '''
        is_str = np.array(map(lambda x: type(x) == str, list(df.value)),
                          dtype=np.bool)
        idx = df.valueuom.fillna('').apply(
            lambda s: 'torr' not in s.lower()) & (
                      is_str | (~is_str & (v > 1.0)))

        v.loc[idx] = v[idx] / 100.
        return v


# GLUCOSE, PH: sometimes have ERROR as value
def clean_lab(df):
    v = df.value.copy()
    idx = v.apply(
        lambda s: type(s) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', s))
    v.loc[idx] = np.nan
    return v.astype(float)


# O2SAT: small number of 0<x<=1 that should be mapped to 0-100 scale
def clean_o2sat(df):
    # change "ERROR" to NaN
    v = df.value.copy()
    idx = v.apply(
        lambda s: type(s) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', s))
    v.loc[idx] = np.nan

    v = v.astype(float)
    idx = (v <= 1)
    v.loc[idx] = v[idx] * 100.
    return v
