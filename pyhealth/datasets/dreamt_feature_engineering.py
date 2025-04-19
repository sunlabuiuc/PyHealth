"""
This module provides a set of helper functions for performing feature engineering using the DREAMT dataset's aggregated E4 dataframes. 

Reference Code Repo:
https://github.com/WillKeWang/DREAMT_FE

Paper:
Addressing Wearable Sleep Tracking Inequity: A New Dataset and Novel Methods for a Population with Sleep Disorders
https://raw.githubusercontent.com/mlresearch/v248/main/assets/wang24a/wang24a.pdf


Main Functions:
- extract_domain_features: Extracts domain features from the E4 dataframes.
- test_fe_all_subjects: Tests the extract_domain_features function on all participants in the DREAMT dataset.

Usage:
To use these functions, import this script and call the desired function with the appropriate parameters. 

For example:

from feature_engineering import *
fg
fe_df = extract_domain_features(data, features, window_size=10)

# this function computes features for all participants
test_fe_all_subjects()


Author: 
License: 
"""

import numpy as np
import pandas as pd
import warnings
import os
from scipy import signal
from multiprocessing import Pool
from tqdm import tqdm

warnings.filterwarnings("ignore")
from aeon.datasets import load_classification
from sklearn.preprocessing import LabelEncoder
from scipy.signal import find_peaks_cwt
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import skew
from scipy.interpolate import interp1d
from scipy.stats import skew

from sklearn.metrics import f1_score, cohen_kappa_score

import lightgbm as lgb
import gpboost as gpb
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from imblearn.over_sampling import SMOTE

from torch.utils.data import DataLoader

import neurokit2 as nk
import heartpy as hp

from collections import Counter

import warnings

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score, auc, precision_recall_curve, roc_auc_score, cohen_kappa_score
from sklearn.base import TransformerMixin

# from scipy.signal import gaussian, convolve, windows -- Guassian is now in .windows path
from scipy.signal import convolve, windows
from scipy.signal.windows import gaussian
from scipy.ndimage import gaussian_filter1d

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

np.random.seed(1)

# define global variables, for the DREAMT dataset
ALL_LABELS = [
    "Sleep_Stage",
    "Obstructive_Apnea",
    "Central_Apnea",
    "Hypopnea",
    "Multiple_Events",
]

ALL_SIGNALS = [
    "TIMESTAMP",
    "BVP",
    "IBI",
    "EDA",
    "TEMP",
    "ACC_X",
    "ACC_Y",
    "ACC_Z",
    "HR",
]

HRV_feature_names = [
    "PPG_Rate_Mean",
    "HRV_MeanNN",
    "HRV_SDNN",
    "HRV_RMSSD",
    "HRV_SDSD",
    "HRV_CVNN",
    "HRV_CVSD",
    "HRV_MedianNN",
    "HRV_MadNN",
    "HRV_MCVNN",
    "HRV_IQRNN",
    "HRV_SDRMSSD",
    "HRV_Prc20NN",
    "HRV_Prc80NN",
    "HRV_pNN50",
    "HRV_pNN20",
    "HRV_MinNN",
    "HRV_MaxNN",
    "HRV_HTI",
    "HRV_TINN",
    "HRV_LF",
    "HRV_HF",
    "HRV_VHF",
    "HRV_TP",
    "HRV_LFHF",
    "HRV_LFn",
    "HRV_HFn",
    "HRV_LnHF",
    "HRV_SD1",
    "HRV_SD2",
    "HRV_SD1SD2",
    "HRV_S",
    "HRV_CSI",
    "HRV_CVI",
    "HRV_CSI_Modified",
    "HRV_PIP",
    "HRV_IALS",
    "HRV_PSS",
    "HRV_PAS",
    "HRV_GI",
    "HRV_SI",
    "HRV_AI",
    "HRV_PI",
    "HRV_C1d",
    "HRV_C1a",
    "HRV_SD1d",
    "HRV_SD1a",
    "HRV_C2d",
    "HRV_C2a",
    "HRV_SD2d",
    "HRV_SD2a",
    "HRV_Cd",
    "HRV_Ca",
    "HRV_SDNNd",
    "HRV_SDNNa",
    "HRV_DFA_alpha1",
    "HRV_MFDFA_alpha1_Width",
    "HRV_MFDFA_alpha1_Peak",
    "HRV_MFDFA_alpha1_Mean",
    "HRV_MFDFA_alpha1_Max",
    "HRV_MFDFA_alpha1_Delta",
    "HRV_MFDFA_alpha1_Asymmetry",
    "HRV_MFDFA_alpha1_Fluctuation",
    "HRV_MFDFA_alpha1_Increment",
    "HRV_ApEn",
    "HRV_SampEn",
    "HRV_ShanEn",
    "HRV_FuzzyEn",
    "HRV_MSEn",
    "HRV_CMSEn",
    "HRV_RCMSEn",
    "HRV_CD",
    "HRV_HFD",
    "HRV_KFD",
    "HRV_LZC",
]


def exclude_signal(segment_df, plot=False, print_scores=False):
    """Identify is a segment is to be labeled as 'artifact'

    Parameters
    ----------
    segment_df: pd.DataFrame 
        a pandas dataframe of a segment of E4 data from 30 seconds.
    plot: bool, optional 
        whether to plot the signal or not. Defaults to False.
    print_scores: bool, optional
        whether you which the scores to be printed. Defaults to False.

    Returns
    -------
    integer
        0 = not artifact, 1 = artifact
    """

    bvp = segment_df.BVP.to_numpy()
    bvp = bvp - np.mean(bvp)

    # Filter specifications
    lowcut = 0.5
    highcut = 15
    fs = 64  # Sampling frequency in Hz for the BVP signal
    b, a = butter(N=2, Wn=[lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype="band")
    filtered_signal = filtfilt(b, a, bvp)

    # Calculate Signal Power
    signal_power = np.mean(filtered_signal**2)

    # Noise Power estimation (assuming you have a noise segment or method)
    noise = bvp - filtered_signal
    noise_power = np.mean(noise**2)

    # Calculate SNR
    snr_db = 10 * np.log10(signal_power / noise_power)

    # Accelometer signal preprocessing for artifact detection
    acc_freq = 32
    bvp_freq = 64

    # actigraphy preprocessing
    acc_x = (segment_df.ACC_X.to_numpy() / 64)[::2]
    acc_y = (segment_df.ACC_Y.to_numpy() / 64)[::2]
    acc_z = (segment_df.ACC_Z.to_numpy() / 64)[::2]

    # interpolate the signal to match the BVP signal
    original_length = len(acc_x)
    original_time_scale = np.linspace(
        0, original_length / acc_freq, original_length, endpoint=True
    )

    if acc_x.shape[0]*2 == bvp.shape[0]:
        new_time_scale = np.linspace(
            0,
            int(original_length / acc_freq),
            int(original_length * (bvp_freq / acc_freq)),
            endpoint=False,
        )
    else:
        new_time_scale = np.linspace(
            0,
            int(original_length / acc_freq),
            int(original_length * (bvp_freq / acc_freq)),
            endpoint=False,
        )

    if new_time_scale.shape[0] >= segment_df.shape[0]:
        acc_x = interp1d(original_time_scale, acc_x, kind="linear")(new_time_scale)[
            : segment_df.shape[0]
        ]
        acc_y = interp1d(original_time_scale, acc_y, kind="linear")(new_time_scale)[
            : segment_df.shape[0]
        ]
        acc_z = interp1d(original_time_scale, acc_z, kind="linear")(new_time_scale)[
            : segment_df.shape[0]
        ]
    else:
        # pad
        pad_length = segment_df.shape[0] - new_time_scale.shape[0]
        acc_x = np.pad(acc_x, (0, pad_length), "edge")
        acc_y = np.pad(acc_y, (0, pad_length), "edge")
        acc_z = np.pad(acc_z, (0, pad_length), "edge")

    # calculate the accelerometer summary statistics, acc_std
    acc = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    acc_epoch_length = 64 * 5
    num_complete_periods = len(acc) // (acc_epoch_length)

    # Reshape the array to ignore incomplete period
    reshaped_acc = acc[: num_complete_periods * acc_epoch_length].reshape(
        (num_complete_periods, -1)
    )

    acc_std = np.std(reshaped_acc, axis=1)[0]

    if plot == True:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        ax1.plot(bvp)
        ax1.plot(filtered_signal)
        ax2.plot(acc)

        plt.show()

    if print_scores == True:
        print("acc_std = {}".format(acc_std))
        print("snr_db = {}".format(snr_db))
        print("bvp_min = {}".format(np.min(bvp)))
        print("bvp_max = {}".format(np.max(bvp)))

    # rule-based artifact detection: return 1 if the segment is an artifact, 0 otherwise
    if acc_std >= 0.4125 / 2 or snr_db < 10 or np.max(bvp) > 500 or np.min(bvp) < -500:
        return 1
    else:
        return 0


def circadian_cosine(ts, samp_freq=64):
    """
    Use the timestamp information to caluclate circadian related features,
    with a cosine function.

    Parameters
    ----------
    ts : numpy.ndarray
        time series array of timestamps.

    Returns
    -------
    numpy.ndarray
        time series array of circadian information in a cosine function.
    """
    start_timestamp = ts[0]
    end_timestamp = ts[ts.shape[0] - 1]
    len_vec = (end_timestamp - start_timestamp) * samp_freq
    in_array = np.arange(0, len_vec + 1)
    out_array = np.sin((2 * np.pi / (len_vec * 2)) * in_array)
    return out_array[: ts.shape[0]]


def circadian_decay(ts, samp_freq=64):
    """
    Use the timestamp information to caluclate circadian related features,
    with a decay model, or exponential decay function.

    Parameters
    ----------
    ts : numpy.ndarray
        time series array of timestamps.

    Returns
    -------
    numpy.ndarray
        time series array of circadian information in a decay function.
    """
    start_timestamp = ts[0]
    end_timestamp = ts[ts.shape[0] - 1]
    len_vec = (end_timestamp - start_timestamp) * samp_freq
    in_array = np.arange(0, len_vec + 1)
    k = np.log(0.01) / len_vec
    out_array = np.exp(k * in_array)
    return out_array[: ts.shape[0]]


def circadian_linear(ts, samp_freq=64):
    """
    Use the timestamp information to caluclate circadian related features,
    with a linear function.

    Parameters
    ----------
    ts : numpy.ndarray
        time series array of timestamps.

    Returns
    -------
    numpy.ndarray
        time series array of circadian information in a linear function.
    """
    start_timestamp = ts[0]
    end_timestamp = ts[ts.shape[0] - 1]
    len_vec = (end_timestamp - start_timestamp) * samp_freq
    in_array = np.arange(0, len_vec + 1)
    out_array = in_array / (max(in_array))
    return out_array[: ts.shape[0]]


def preprocess_BVP(bvp):
    """
    Preprocess a time series of blood volume pulse signal from Empatica E4.

    Parameters
    ----------
    bvp : numpy.ndarray
        time series array of Blood Volume Pulse.

    Returns
    -------
    numpy.ndarray
        time series array of filtered blood volume pulse signal.
    """

    # define low and high frequency cut-off
    low = 0.5
    high = 15

    # chebyshev filter
    sos = signal.cheby2(
        N=10, rs=40, Wn=[low, high], btype="bandpass", fs=64, output="sos"
    )
    bvp_filtered = signal.sosfilt(sos, bvp)
    return bvp_filtered


def preprocess_ACC(acc):
    """
    Preprocess an array of accelerometry signal coming from Empatica E4.

    Parameters
    ----------
    acc : numpy.ndarray
        time series array of an accel.erometry signal.

    Returns
    -------
    numpy.ndarray
        time series array of the preprocessed accelerometry signal.
    """
    acc_sos = signal.butter(N=3, Wn=[3, 10], btype="bp", fs=32, output="sos")
    acc_filtered = signal.sosfilt(acc_sos, acc)
    return acc_filtered


def preprocess_EDA(eda):
    """
    Preprocess an array of electrodermal activity signal coming from Empatica E4.

    Parameters
    ----------
    eda : numpy.ndarray
        time series array of an electrodernal activity signal.

    Returns
    -------
    numpy.ndarray
        time series array of the preprocessed electrodermal activity signal.
    """
    total_length = eda.shape[0]
    # downsample to reflect actual frequency
    eda = eda[1::16]

    # detrend the eda signal at every 5-second interval
    x = np.arange(0, 20, 1)
    detrended = []
    for i in range((len(eda) // 20)):
        segment = eda[(i * 20) : (20 * (i + 1))]
        m, b = np.polyfit(x, segment, 1)
        d = segment - x * m - b
        detrended.append(d)

    # find the detrended value for the remaining segments (less than 5s)
    last = eda[(len(eda) // 20 * 20) : len(eda)]
    if len(last) >= 10:
        last_x = np.arange(0, len(last), 1)
        m, b = np.polyfit(last_x, last, 1)
        last_d = last - last_x * m - b
        detrended.append(last_d)
    elif 10 > len(last) > 0:
        last_d = last - np.mean(last)
        detrended.append(last_d)
    else:
        pass

    detrended_2 = [d.tolist() for d in detrended]
    detrended_EDA = [d for sublist in detrended_2 for d in sublist]
    eda = detrended_EDA

    # filtering of the detrended signals
    order = 3
    frequency = 0.7
    LPF = signal.butter(N=order, Wn=frequency, fs=4, output="sos")
    filt_rEDA = signal.sosfilt(LPF, eda)

    # repeat the signals so that it has the same length as the high frequency bvp signal
    filt_rEDA_repeated = np.repeat(filt_rEDA, 16)
    if filt_rEDA_repeated.shape[0] > total_length:
        filt_rEDA_repeated = filt_rEDA_repeated[:total_length]
    elif filt_rEDA_repeated.shape[0] < total_length:
        filt_rEDA_repeated = np.pad(
            filt_rEDA_repeated,
            ((total_length - filt_rEDA_repeated.shape[0]), 0),
            "mean",
        )

    return filt_rEDA_repeated


def clean_IBI(df, freq=64):
    """
    Preprocess the Inter-Beat Interval (IBI) signal from Empatica E4.

    Parameters
    ----------
    df : pandas.DataFrame
        E4 dataframe with IBI signal.
    freq : int, optional
        Sampling frequency of the IBI signal. Defaults to 64.

    Returns
    -------
    pandas.DataFrame
        E4 dataframe with the preprocessed IBI signal.
    """

    def detect_motion_artifact(df, window_seconds=10, freq=64):
        window = window_seconds * freq
        acc_x = df.ACC_X.to_numpy()
        acc_x_diff = np.diff(acc_x)
        acc_y = df.ACC_Y.to_numpy()
        acc_y_diff = np.diff(acc_y)
        acc_z = df.ACC_Z.to_numpy()
        acc_z_diff = np.diff(acc_z)

        ma_flag_array = np.zeros((acc_x.shape[0],))
        for i in range(1, acc_x.shape[0]):
            acc_x_diff_window = acc_x_diff[i : (i + window)]
            acc_y_diff_window = acc_y_diff[i : (i + window)]
            acc_z_diff_window = acc_z_diff[i : (i + window)]

            if (
                np.sum(np.abs(acc_x_diff_window) > 5) >= 5
                or np.sum(np.abs(acc_y_diff_window) > 5) >= 5
                or np.sum(np.abs(acc_z_diff_window) > 5) >= 5
            ):
                ma_flag_array[i] = 1

        return ma_flag_array

    def filter_close_peaks(peaks, min_distance):
        filtered_peaks = [peaks[0]]
        for peak in peaks[1:]:
            if peak - filtered_peaks[-1] > min_distance:
                filtered_peaks.append(peak)
        return np.array(filtered_peaks)

    def process_chunk(chunk):
        """Calculate HR for every second
        Args: chunk (np.ndarray): numpoy array
        Returns: preprocessed chunk
        """
        mean = np.mean(chunk)
        inverse_mean = 60 / (mean + 1e-10)
        return np.full(chunk.shape, inverse_mean)

    bvp = df.BVP.to_numpy()

    # detect peaks
    widths = np.arange(2, 32)
    peaks = find_peaks_cwt(bvp, widths, window_size=16)
    peaks = filter_close_peaks(peaks, min_distance=12)

    segment_length = bvp.shape[0]
    ibi_array = np.diff(peaks)
    ibi_array = np.insert(ibi_array, 0, peaks[0])
    repeat_counts = np.insert(
        ibi_array[1:], ibi_array.shape[0] - 1, segment_length - peaks[-1]
    )
    repeated_array = np.repeat(ibi_array, repeat_counts)

    # Calculate how many zeros to add at the front
    pad_width = segment_length - len(repeated_array)
    if pad_width < 0:
        raise ValueError("Final length must be greater than the current array length.")

    # Pad the array
    padded_array = np.pad(
        repeated_array, (pad_width, 0), mode="constant", constant_values=0
    )

    cleaned_ibi_array = padded_array / freq
    motion_artifact_array = detect_motion_artifact(df)

    df["IBI"] = cleaned_ibi_array
    df["motion_artifact"] = motion_artifact_array

    chunk_size = 64  # corresponds to 1 Hz
    inverse_mean_repeated = np.concatenate(
        [
            process_chunk(cleaned_ibi_array[i : i + chunk_size])
            for i in range(0, len(cleaned_ibi_array), chunk_size)
        ]
    )

    df["HR"] = inverse_mean_repeated

    return df


def preprocess_TIMESTAMPS(ts):
    """
    Obtain all three circadian information

    Parameters
    ----------
    ts : numpy.ndarray
        time series array of timestamps.

    Returns
    -------
    ts_cos : numpy.ndarray
        time series array of circadian information in a cosine function.
    ts_dec : numpy.ndarray
        time series array of circadian information in a decay function.
    ts_lin : numpy.ndarray
        time series array of circadian information in a linear function.
    """
    ts_cos = circadian_cosine(ts, samp_freq=64)
    ts_dec = circadian_decay(ts, samp_freq=64)
    ts_lin = circadian_linear(ts, samp_freq=64)
    return ts_cos, ts_dec, ts_lin


def preprocess_ALL_SIGNALS(df):
    """
    Preprocess all available signals from a dataset from Empatica E4

    Parameters
    ----------
    df : pandas.DataFrame
        pandas dataframe of E4 data.

    Returns
    -------
    pandas.DataFrame
        preprocessed E4 dataframe
    """
    # initialize preprocessed_df with all available labels (outcomes)
    if df.shape[0] > 12*3600*64:
        df = df.iloc[:12*3600*64, :]
    preprocessed_df = df.loc[:12*3600*64, ALL_LABELS]

    # perform preprocessing
    ts = df['TIMESTAMP'].to_numpy()
    ts = ts - ts[0]
    eda = preprocess_EDA(df.EDA.to_numpy())
    acc_x = preprocess_ACC(df.ACC_X.to_numpy())
    acc_y = preprocess_ACC(df.ACC_Y.to_numpy())
    acc_z = preprocess_ACC(df.ACC_Z.to_numpy())
    bvp = preprocess_BVP(df.BVP.to_numpy())
    ts_cos, ts_dec, ts_lin = preprocess_TIMESTAMPS(ts)

    # save preprocessed signals
    preprocessed_df["HR"] = df.HR
    preprocessed_df["TEMP"] = df.TEMP
    preprocessed_df["TIMESTAMP"] = ts
    preprocessed_df["TIMESTAMP_COSINE"] = ts_cos
    preprocessed_df["TIMESTAMP_DECAY"] = ts_dec
    preprocessed_df["TIMESTAMP_LINEAR"] = ts_lin
    preprocessed_df["ACC_X"] = acc_x
    preprocessed_df["ACC_Y"] = acc_y
    preprocessed_df["ACC_Z"] = acc_z
    preprocessed_df["EDA"] = eda
    preprocessed_df["BVP"] = bvp
    preprocessed_df["IBI"] = df.IBI

    preprocessed_df = clean_IBI(preprocessed_df)

    return preprocessed_df


def acc_trimmed_summary(acc):
    """
    Calculate mean, max, and IQR of the trimmed accelerometer signal

    Parameters
    ----------
    acc : numpy.ndarray
        time series array of an accelerometer signal.

    Returns
    -------
    (numpy.float64, numpy.float64, numpy.float64)
        mean, max, and IQR of the trimmed accelerometer signal
    """
    acc_filtered = acc[(acc > np.quantile(acc, 0.10)) & (acc < np.quantile(acc, 0.90))]
    if acc_filtered.shape[0] == 0:
        return (
            np.mean(acc),
            np.max(acc),
            0,
        )
    else:
        return (
            np.mean(acc_filtered),
            np.max(acc_filtered),
            np.quantile(acc_filtered, 0.75) - np.quantile(acc, 0.25),
        )


def MAD_trimmed_summary(acc, segment_seconds=30):
    """
    Calculate mean, max, and IQR of the mean amplitude deviation (MAD) accelerometer signal

    Parameters
    ----------
    acc : numpy.ndarray
        time series array of an accelerometer signal.
    segment_seconds : int, optional
        length of the segment in seconds. Defaults to 30.

    Returns
    -------
    (numpy.float64, numpy.float64, numpy.float64)
        mean, max, and IQR of the MAD accelerometer signal
    """

    acc_filtered = acc[(acc > np.quantile(acc, 0.10)) & (acc < np.quantile(acc, 0.90))]
    if acc_filtered.shape[0] == 0:
        return (
            np.mean(acc),
            np.max(acc),
            0,
        )
    else:
        # split into 5 second sub-arrays:
        splits = np.array_split(acc, int(segment_seconds / 5))
        # Calculate MAD on every sub array:
        MADs = [np.mean(np.abs(split - np.mean(split))) for split in splits]
        return (
            np.mean(MADs),
            np.max(MADs),
            np.quantile(MADs, 0.75) - np.quantile(MADs, 0.25),
        )


def ACC_summary(segment_df):
    """
    compute summary statistics for the accelerometer signal

    Parameters
    ----------
    segment_df : pd.DataFrame
        a pandas dataframe of a segment of E4 data from 30 seconds.

    Returns
    -------
    dict
        a dictionary of accelerometer summary statistics
    """
    acc_x = (segment_df.ACC_X.to_numpy() / 64)[::2]
    acc_x = np.clip(acc_x, np.min(acc_x), np.max(acc_x))
    acc_y = (segment_df.ACC_Y.to_numpy() / 64)[::2]
    acc_y = np.clip(acc_y, np.min(acc_y), np.max(acc_y))
    acc_z = (segment_df.ACC_Z.to_numpy() / 64)[::2]
    acc_z = np.clip(acc_z, np.min(acc_z), np.max(acc_z))
    (
        ACC_X_trimmed_mean,
        ACC_X_trimmed_max,
        ACC_X_trimmed_IQR,
    ) = acc_trimmed_summary(acc_x)
    (
        ACC_Y_trimmed_mean,
        ACC_Y_trimmed_max,
        ACC_Y_trimmed_IQR,
    ) = acc_trimmed_summary(acc_y)
    (
        ACC_Z_trimmed_mean,
        ACC_Z_trimmed_max,
        ACC_Z_trimmed_IQR,
    ) = acc_trimmed_summary(acc_z)
    (
        ACC_X_MAD_trimmed_mean,
        ACC_X_MAD_trimmed_max,
        ACC_X_MAD_trimmed_IQR,
    ) = MAD_trimmed_summary(acc_x)
    (
        ACC_Y_MAD_trimmed_mean,
        ACC_Y_MAD_trimmed_max,
        ACC_Y_MAD_trimmed_IQR,
    ) = MAD_trimmed_summary(acc_y)
    (
        ACC_Z_MAD_trimmed_mean,
        ACC_Z_MAD_trimmed_max,
        ACC_Z_MAD_trimmed_IQR,
    ) = MAD_trimmed_summary(acc_z)

    acc_freq = 32
    bvp_freq = 64

    # Calculate original length and time scales
    original_length = len(acc_x)
    original_time_scale = np.linspace(
        0, original_length / acc_freq, original_length, endpoint=False
    )

    # Calculate new time scale
    new_length = int(original_length * (bvp_freq / acc_freq))
    new_time_scale = np.linspace(0, original_time_scale[-1], new_length, endpoint=False)

    # Interpolate the signal to match the BVP signal
    if new_time_scale.shape[0] >= segment_df.shape[0]:
        acc_x = interp1d(original_time_scale, acc_x, kind="linear")(new_time_scale)[
            : segment_df.shape[0]
        ]
        acc_y = interp1d(original_time_scale, acc_y, kind="linear")(new_time_scale)[
            : segment_df.shape[0]
        ]
        acc_z = interp1d(original_time_scale, acc_z, kind="linear")(new_time_scale)[
            : segment_df.shape[0]
        ]
    else:
        # pad
        pad_length = segment_df.shape[0] - new_time_scale.shape[0]
        acc_x = np.pad(acc_x, (0, pad_length), "edge")
        acc_y = np.pad(acc_y, (0, pad_length), "edge")
        acc_z = np.pad(acc_z, (0, pad_length), "edge")

    # Calculate the accelerometer summary statistics, acc_std
    acc = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    acc_epoch_length = 64 * 5
    num_complete_periods = len(acc) // (acc_epoch_length)
    reshaped_acc = acc[: num_complete_periods * acc_epoch_length].reshape(
        (num_complete_periods, -1)
    )
    acc_stds = np.std(reshaped_acc, axis=1)

    # activity index for every minute
    reshaped_acc_std = acc_stds.reshape((6, -1))
    acc_ind = np.sum(reshaped_acc_std, axis=0)[0]

    acc_features_dict = {
        "ACC_X_trimmed_mean": ACC_X_trimmed_mean,
        "ACC_X_trimmed_max": ACC_X_trimmed_max,
        "ACC_X_trimmed_IQR": ACC_X_trimmed_IQR,
        "ACC_Y_trimmed_mean": ACC_Y_trimmed_mean,
        "ACC_Y_trimmed_max": ACC_Y_trimmed_max,
        "ACC_Y_trimmed_IQR": ACC_Y_trimmed_IQR,
        "ACC_Z_trimmed_mean": ACC_Z_trimmed_mean,
        "ACC_Z_trimmed_max": ACC_Z_trimmed_max,
        "ACC_Z_trimmed_IQR": ACC_Z_trimmed_IQR,
        "ACC_X_MAD_trimmed_mean": ACC_X_MAD_trimmed_mean,
        "ACC_X_MAD_trimmed_max": ACC_X_MAD_trimmed_max,
        "ACC_X_MAD_trimmed_IQR": ACC_X_MAD_trimmed_IQR,
        "ACC_Y_MAD_trimmed_mean": ACC_Y_MAD_trimmed_mean,
        "ACC_Y_MAD_trimmed_max": ACC_Y_MAD_trimmed_max,
        "ACC_Y_MAD_trimmed_IQR": ACC_Y_MAD_trimmed_IQR,
        "ACC_Z_MAD_trimmed_mean": ACC_Z_MAD_trimmed_mean,
        "ACC_Z_MAD_trimmed_max": ACC_Z_MAD_trimmed_max,
        "ACC_Z_MAD_trimmed_IQR": ACC_Z_MAD_trimmed_IQR,
        "ACC_INDEX": acc_ind,
    }

    return acc_features_dict


def TEMP_summary(segment_df):
    """
    Compute summary statistics for the temperature signal

    Parameters
    ----------
    segment_df: pd.DataFrame
        a pandas dataframe of a segment of E4 data from 30 seconds.

    Note
    ----
    The temperature signal is expected to be in degrees Celsius. Excludes temperatures lower than 28 and higher than 40

    Returns
    -------
    dict
        a dictionary of temperature summary statistics
    """

    """Preprocesses Temperature data
    
    returns mean max min and standard devation"""
    temps = segment_df.loc[:, "TEMP"].to_numpy()
    temps = temps[(temps >= 31) & (temps <= 40)]
    try:
        temp_features_dict = {
            "TEMP_mean": np.mean(temps),
            "TEMP_median": np.median(temps),
            "TEMP_max": np.max(temps),
            "TEMP_min": np.min(temps),
            "TEMP_std": np.std(temps),
        }
    except:
        temp_features_dict = {
            "TEMP_mean": np.nan,
            "TEMP_median": np.nan,
            "TEMP_max": np.nan,
            "TEMP_min": np.nan,
            "TEMP_std": np.nan,
        }
    return temp_features_dict


def HRV_summary(segment_df, segment_seconds=30, freq=64):
    """
    Compute summary statistics for the Heart Rate Variability (HRV) signal

    Parameters
    ----------
    segment_df: pd.DataFrame
        a pandas dataframe of a segment of E4 data from 30 seconds.

    Returns
    -------
    dict
        a dictionary of HRV summary statistics
    """

    bvp = segment_df.loc[:, "BVP"].to_numpy()
    ibi = segment_df.loc[:, "IBI"].to_numpy()
    HRV_features_dict = {k: np.nan for k in HRV_feature_names}

    # New - Find peak of spectrum - Code from above
    N = bvp.shape[0]
    T = 1.0 / 64.0
    yf = np.fft.fft(bvp)
    xf = np.fft.fftfreq(N, T)[: N // 2]

    fft_magnitudes = 2.0 / N * np.abs(yf[0 : N // 2])
    total_power = np.sum(ibi**2)
    normalized_ibi = (ibi - np.min(ibi)) / (np.max(ibi) - np.min(ibi))
    normalized_power = np.sum(normalized_ibi**2)
    xLF = xf[(xf >= 0.04) & (xf < 0.15)]
    mLF = fft_magnitudes[(xf >= 0.04) & (xf < 0.15)]
    xHF = xf[(xf >= 0.15) & (xf < 0.4)]
    mHF = fft_magnitudes[(xf >= 0.15) & (xf < 0.4)]

    # LF peak frequency
    peak_LF = np.where(mLF == np.max(mLF))[0]
    LF_peak_frequency = xf[peak_LF[0]]

    # HF peak frequency
    peak_HF = np.where((mHF == np.max(mHF)))[0]
    HF_peak_frequency = xf[peak_HF[0]]

    signals, info = nk.ppg_process(bvp, sampling_rate=freq)

    error_code = 0

    # if bvp.shape[0] != segment_seconds * freq:
    #     print(bvp.shape)
    try:
        results = nk.ppg_analyze(signals, sampling_rate=freq)
        for hrv_fn in HRV_feature_names:
            HRV_features_dict[hrv_fn] = results.loc[0, hrv_fn]
        HRV_features_dict["total_power"] = total_power
        HRV_features_dict["normalized_power"] = normalized_power
        HRV_features_dict["LF_frequency_power"] = results.loc[0, "HRV_LF"] * total_power
        HRV_features_dict["HF_frequency_power"] = results.loc[0, "HRV_HF"] * total_power
        HRV_features_dict["LF_frequency_peak"] = LF_peak_frequency
        HRV_features_dict["HF_frequency_peak"] = HF_peak_frequency
        HRV_features_dict["LF_normalized_power"] = (
            results.loc[0, "HRV_LFn"] * normalized_power
        )
        HRV_features_dict["HF_normalized_power"] = (
            results.loc[0, "HRV_HFn"] * normalized_power
        )
    except:
        error_code = -1

    HRV_features_dict["breathing_rate"] = HF_peak_frequency * 60

    HRV_features_dict["HR_mean"] = segment_df.HR.mean()
    HRV_features_dict["HR_median"] = segment_df.HR.median()
    HRV_features_dict["HR_max"] = segment_df.HR.max()
    HRV_features_dict["HR_min"] = segment_df.HR.min()
    HRV_features_dict["HR_range"] = segment_df.HR.max() - segment_df.HR.min()
    HRV_features_dict["HR_std"] = segment_df.HR.std()

    HRV_features_dict["BVP_mean"] = segment_df.BVP.mean()
    HRV_features_dict["BVP_median"] = segment_df.BVP.median()
    HRV_features_dict["BVP_max"] = segment_df.BVP.max()
    HRV_features_dict["BVP_min"] = segment_df.BVP.min()
    HRV_features_dict["BVP_range"] = segment_df.BVP.max() - segment_df.BVP.min()
    HRV_features_dict["BVP_std"] = segment_df.BVP.std()

    return HRV_features_dict, error_code


def EDA_summary(segment_df):
    """
    Compute summary statistics for the Electrodermal Activity (EDA) signal

    Parameters
    ----------
    segment_df: pd.DataFrame   
        a pandas dataframe of a segment of E4 data from 30 seconds.

    Returns
    -------
    dict
        a dictionary of EDA summary statistics
    """

    eda = segment_df.loc[:, "EDA"].to_numpy()
    sampling_rate = 4
    eda_signal = nk.signal_sanitize(eda)

    # Series check for non-default index
    if type(eda_signal) is pd.Series and type(eda_signal.index) != pd.RangeIndex:
        eda_signal = eda_signal.reset_index(drop=True)

    # Preprocess
    eda_cleaned = eda_signal  # Add your custom cleaning module here or skip cleaning
    eda_decomposed = nk.eda_phasic(eda_cleaned, sampling_rate=sampling_rate)

    # Find peaks
    try:
        peak_signal, info = nk.eda_peaks(
            eda_decomposed["EDA_Phasic"].values,
            sampling_rate=sampling_rate,
            method="neurokit",
            amplitude_min=0.1,
        )
        info["sampling_rate"] = sampling_rate  # Add sampling rate in dict info
    except:
        # print("{} has EDA peak detection error".format(self.sid))
        return {
            "mean_SCR_Height": np.nan,
            "mean_SCR_Amplitude": np.nan,
            "mean_SCR_RiseTime": np.nan,
            "mean_SCR_RecoveryTime": np.nan,
            "max_SCR_Height": np.nan,
            "max_SCR_Amplitude": np.nan,
            "max_SCR_RiseTime": np.nan,
            "max_SCR_RecoveryTime": np.nan,
        }

    signals = pd.DataFrame({"EDA_Raw": eda_signal, "EDA_Clean": eda_cleaned})
    signals = pd.concat([signals, eda_decomposed, peak_signal], axis=1)

    SCR_names = [
        "SCR_Height",
        "SCR_Amplitude",
        "SCR_RiseTime",
        "SCR_RecoveryTime",
    ]
    features_dict = {}
    try:
        for k in SCR_names:
            x = info[k]
            x = x[~np.isnan(x)]
            features_dict["".join(["mean_", k])] = np.mean(x)
            features_dict["".join(["max_", k])] = np.max(x)
    except:
        features_dict = {
            "mean_SCR_Height": np.nan,
            "mean_SCR_Amplitude": np.nan,
            "mean_SCR_RiseTime": np.nan,
            "mean_SCR_RecoveryTime": np.nan,
            "max_SCR_Height": np.nan,
            "max_SCR_Amplitude": np.nan,
            "max_SCR_RiseTime": np.nan,
            "max_SCR_RecoveryTime": np.nan,
        }

    return features_dict


def circadian_features(segment_df):
    """
    Compute circadian features from the timestamp information
    
    Parameters
    ----------
    segment_df: pd.DataFrame
        a pandas dataframe of a segment of E4 data from 30 seconds.

    Returns
    -------
    dict
        a dictionary of circadian features
    """
    circadian_features_dict = {
        "timestamp_start": segment_df.loc[0, "TIMESTAMP"],
        "circadian_cosine": segment_df.loc[0, "TIMESTAMP_COSINE"],
        "circadian_decay": segment_df.loc[0, "TIMESTAMP_DECAY"],
        "circadian_linear": segment_df.loc[0, "TIMESTAMP_LINEAR"],
    }
    return circadian_features_dict


def aggregate_labels(segment_df, segment_seconds=30):
    """
    Aggregate labels for the segment

    Parameters
    ----------
    segment_df: pd.DataFrame
        a pandas dataframe of a segment of E4 data from 30 seconds.
    segment_seconds: int, optional
        length of the segment in seconds. Defaults to 30.

    Returns
    -------
    dict
        a dictionary of aggregated labels
    """
    segment_df.loc[:, ALL_LABELS] = segment_df.loc[:, ALL_LABELS].fillna(0)

    # a one-liner function that takes a NumPy array and returns the value with more than 50% occurrence,
    # or "Mixed" if no such value exists, using a combination of np.unique and a list comprehension

    def most_common_or_mixed(arr):
        unique, counts = np.unique(arr, return_counts=True)
        if any(counts > len(arr) / 2):
            return unique[counts > len(arr) / 2][0]
        else:
            return "Mixed"

    if segment_seconds == 30:
        num_uniques = np.unique(segment_df.Sleep_Stage).shape[0]
        assert num_uniques == 1

        labels_dict = {"Sleep_Stage": segment_df.Sleep_Stage[0]}
    else:
        labels_dict = {"Sleep_Stage": most_common_or_mixed(segment_df.Sleep_Stage)}

    labels_dict.update({"Obstructive_Apnea": np.max(segment_df.Obstructive_Apnea)})
    labels_dict.update({"Central_Apnea": np.max(segment_df.Central_Apnea)})
    labels_dict.update({"Hypopnea": np.max(segment_df.Hypopnea)})
    labels_dict.update({"Multiple_Events": np.max(segment_df.Multiple_Events)})
    return labels_dict


def fe_per_segment(segment_df, segment_seconds=30):
    """
    Compute features for a segment of E4 data

    Parameters
    ----------
    segment_df: pd.DataFrame
        a pandas dataframe of a segment of E4 data from 30 seconds.
    segment_seconds: int, optional 
        length of the segment in seconds. Defaults to 30.

    Returns
    -------
    dict
        a dictionary of all features    
    """
    all_features_dict, error_flag = HRV_summary(segment_df)
    all_features_dict.update(ACC_summary(segment_df))
    all_features_dict.update(TEMP_summary(segment_df))
    all_features_dict.update(EDA_summary(segment_df))
    all_features_dict.update(circadian_features(segment_df))
    all_features_dict.update(
        aggregate_labels(segment_df, segment_seconds=segment_seconds)
    )
    all_features_dict.update({"artifact": exclude_signal(segment_df)})
    return all_features_dict, error_flag


def fe_whole_night(df):
    """
    Compute features for the whole night of E4 data

    Parameters
    ----------
    df: pd.DataFrame
        a pandas dataframe of E4 data for the whole night.

    Returns
    -------
    dict
        a dictionary of all features
    """
    all_features_dict, _ = HRV_summary(df)
    all_features_dict.update(ACC_summary(df))
    all_features_dict.update(TEMP_summary(df))
    all_features_dict.update(EDA_summary(df))
    return all_features_dict


def extract_domain_features(
    sid,
    data_folder="/whole_dfs",
    segment_seconds=30,
    overlap=False,
    save_folder_dir="./fe_dataframes_whole_study/",
):
    """
    Extract domain features for 1 subject, using already aggregated dataset

    Parameters
    ----------
    sid : str
        string of subject id
    segment_seconds : int
        integer indicating the length of unit epoch to extract features from
    overlap: boolean
        whether there should be overlap between windows,
        length of overlap must be 30

    Returns
    -------
    features_df : pandas.DataFrame
        dataframe of domain features per segment, together with labels
    """
    df_dir = "{}/{}_whole_df.csv".format(data_folder, sid)
    print(df_dir)
    # Check if the folder exists
    if not os.path.exists(save_folder_dir):
        # Create the folder
        os.makedirs(save_folder_dir)

    df = pd.read_csv(df_dir)
    df = preprocess_ALL_SIGNALS(df)

    epoch_length = segment_seconds * 64
    # cut the first segments that are not increment of 64*30
    index = (
        np.where(df.Sleep_Stage == "W")[0][0] if np.any(df.Sleep_Stage == "W") else None
    )
    if index is not None:
        length_to_cut = int(index % 1920)
        df = df.iloc[length_to_cut:(length_to_cut+64*12*3600),]
    else:
        pass
    df = df.reset_index(drop=True)

    # start feature extraction
    example_segment = df.iloc[:epoch_length, :]
    example_fe_dict, error_code = fe_per_segment(
        example_segment, segment_seconds=segment_seconds
    )
    if error_code < 0:
        print("First segment has error ")
    # num_errors = 0
    column_names = list(example_fe_dict.keys())
    column_names.append("sid")
    num_segments = int(df.shape[0] / epoch_length)

    domain_features_df = pd.DataFrame(
        columns=column_names, index=np.arange(num_segments)
    )
    offset = 0
    for i in tqdm(range(num_segments)):
        segment_df = df.iloc[
            (i * epoch_length + offset) : ((i + 1) * epoch_length + offset), :
        ]
        segment_df = segment_df.reset_index(drop=True)
        if segment_df.shape[0] == segment_seconds * 64:
            try:
                segment_fe_dict, error_code = fe_per_segment(
                    segment_df, segment_seconds=segment_seconds
                )
            except TypeError:
                list_stages = segment_df.Sleep_Stage.tolist()
                segment_df.Sleep_Stage = segment_df.Sleep_Stage.fillna(0)
                counts = Counter(list_stages)
                offset += counts[list_stages[0]]
                segment_df = df.iloc[
                    (i * epoch_length + offset) : ((i + 1) * epoch_length + offset), :
                ]
                segment_df = segment_df.reset_index(drop=True)
                segment_fe_dict, error_code = fe_per_segment(segment_df)
            except AssertionError:
                list_stages = segment_df.Sleep_Stage.tolist()
                counts = Counter(list_stages)
                offset += counts[list_stages[0]]
                segment_df = df.iloc[
                    (i * epoch_length + offset) : ((i + 1) * epoch_length + offset), :
                ]
                segment_df = segment_df.reset_index(drop=True)
                segment_fe_dict, error_code = fe_per_segment(segment_df)

        domain_features_df.loc[i] = segment_fe_dict

    domain_features_df["sid"] = np.repeat(sid, num_segments)
    domain_features_df.to_csv(
        save_folder_dir + "/{}_domain_features_df.csv".format(sid), index=False
    )

    return domain_features_df


def fe_whole_night_all_sids(info_dir, data_folder,save_folder_dir):
    """
    Compute features for the whole night of E4 data for all subjects
    
    Parameters
    ----------
    info_dir : str
        string of the directory of the participant_info.csv
    data_folder : str
        string of the directory of the aggregated E4 datasets
    save_folder_dir : str
        string of the directory to save the feature engineering dataframe

    Returns
    -------
    fe_df : pandas.DataFrame
        dataframe of domain features for all subjects
    """
    info_df = pd.read_csv(info_dir)
    list_sids = info_df.SID.to_list()
    example_df = pd.read_csv(
        "{}/{}_whole_df.csv".format( data_folder, list_sids[0]))
    
    example_df = preprocess_ALL_SIGNALS(example_df)
    example_fe_dict = fe_whole_night(example_df)
    column_names = list(example_fe_dict.keys())
    column_names.append("sid")
    fe_df = pd.DataFrame(columns=column_names, index=np.arange(len(list_sids)))

    for i, sid in tqdm(enumerate(list_sids)):
        df_dir = "/aggregate_dataset/{}_whole_df.csv".format(
            sid
        )
        df = pd.read_csv(df_dir)
        df = preprocess_ALL_SIGNALS(df)
        fe_dict = fe_whole_night(df)
        fe_df.loc[i] = fe_dict

    """ 
    Normal: An AHI of less than 5 events per hour.
    Mild Sleep Apnea: An AHI of 5 to 14 events per hour.
    Moderate Sleep Apnea: An AHI of 15 to 29 events per hour.
    Severe Sleep Apnea: An AHI of 30 or more events per hour.
    """

    # add medical indices
    fe_df["sid"] = list_sids
    fe_df["AGE"] = info_df["AGE"]
    fe_df["GENDER"] = info_df["GENDER"]
    fe_df["BMI"] = info_df["BMI"]
    fe_df["OAHI"] = info_df["OAHI"]
    fe_df["AHI"] = info_df["AHI"]
    fe_df["Mean_SaO2"] = info_df["Mean_SaO2"]
    fe_df["Arousal Index"] = info_df["Arousal Index"]

    # add AHI index

    fe_df["Normal"] = (info_df["AHI"] < 5).astype(int)
    fe_df["Mild"] = info_df["AHI"].apply(lambda x: 1 if x >= 5 and x <= 14 else 0)
    fe_df["Moderate"] = info_df["AHI"].apply(lambda x: 1 if x >= 15 and x <= 29 else 0)
    fe_df["Severe"] = info_df["AHI"].apply(lambda x: 1 if x >= 30 else 0)

    fe_df.to_csv("{}/all_subjects_fe_df.csv".format(save_folder_dir))

    return fe_df


def test_domain_feature_engineering(sid):
    """
    test the domain feature engineering function

    Parameters
    ----------
    sid : str
        string of the subject id

    Returns
    -------
    int
        0 if the function runs successfully
    """
    extract_domain_features(
        sid, segment_seconds=30, save_folder_dir="./fe_dataframes_whole_study/"
    )
    return 0


def test_fe_all_subjects(info_dir, data_folder, save_folder_dir):
    """
    Test the feature engineering function for all subjects

    Parameters
    ----------
    info_dir : str
        string of the directory of the participant_info.csv
    data_folder : str
        string of the directory of the aggregated E4 datasets
    save_folder_dir : str
        string of the directory to save the feature engineering dataframe

    Returns
    -------
    list
        list of subject ids that have an error
    """
    list_sids = pd.read_csv(info_dir).SID.to_list()
    error_sids = []
    for sid in list_sids:
        print(sid)
        try:
            extract_domain_features(
                sid, data_folder=data_folder, segment_seconds=30, save_folder_dir= save_folder_dir
            )
        except:
            print("ERROR")
            error_sids.append(sid)
    return error_sids



def load_data_to_df(
    threshold, quality_df_dir, info_df, features_dir, nan_feature_names, label_names, circadian_features
):
    """
    Loads and processes feature data from CSV files for subjects meeting a 
    quality score threshold, applying several preprocessing steps including 
    rolling standard deviations, Gaussian filtering, and derivative calculation. 
    The function also classifies subjects based on Apnea-Hypopnea Index (AHI) 
    and Body Mass Index (BMI) into predefined categories.

    Parameters:
    ----------
    threshold : float
        The threshold for the percentage of data excluded based on quality scores. 
        Subjects with quality scores below this threshold are considered for analysis.
    quality_df_dir : str
        A path to the file summarizing the percentage of artifacts of each subject's 
        data calculated from features dataframe
    info_df : pandas.DataFrame
        A DataFrame containing demographic and clinical information for the subjects, 
        indexed by SID.
    features_dir : str
        A path to the folder containing all the features
    nan_feature_names : list of str
        Names of features that should be considered as NaN and excluded from the analysis.
    label_names : list of str
        Names of columns in the data that are considered as labels and should not be 
        treated as features.
    circadian_features : list of str
        Names of features related to circadian rhythms, treated separately from other 
        physiological features.

    Returns:
    -------
    all_subjects_fe_df : pandas dataFrame
        A DataFrame containing the processed features for all subjects meeting the 
        quality threshold.
    good_quality_sids : list of str
        A list of subject IDs that met the quality score threshold and were included 
        in the analysis.
    """
    # load quality scores
    quality_df = pd.read_csv(quality_df_dir)
    good_quality_sids = quality_df.loc[
        quality_df.percentage_excludes < float(threshold), "sid"
    ].to_list()

    # load demographic info
    info_df.index = info_df.SID

    # Read example from one subject for further processing
    path = str(features_dir) + str(info_df.SID[0]) + '_domain_features_df.csv'
    example_df = pd.read_csv(path)

    # select features
    feature_names = [
        f
        for f in example_df.columns.tolist()
        if f not in nan_feature_names + label_names + ["sid"]
    ]

    # select physiological features
    physiological_features = [f for f in feature_names if f not in circadian_features]

    # create dataframe for all the subjects' features
    example_df = rolling_stds(example_df, physiological_features, window_size=10)
    example_df = gaussian_filtering(
        example_df, physiological_features, kernel_size=20, std_dev=100
    )
    example_df = add_derivatives(example_df, physiological_features)

    all_subjects_fe_df = pd.DataFrame(columns=example_df.columns)
    for sid in good_quality_sids:
        path = str(features_dir) + sid + '_domain_features_df.csv'
        sid_df = pd.read_csv(path)
        sid_df = rolling_stds(sid_df, physiological_features, window_size=10)
        sid_df = gaussian_filtering(
            sid_df, physiological_features, kernel_size=20, std_dev=100
        )
        sid_df = add_derivatives(sid_df, physiological_features)

        # add apnea target
        subject_AHI = int(info_df.loc[sid, "AHI"])
        if subject_AHI < 5:
            sid_df["AHI_Severity"] = 0
        elif 5 <= subject_AHI < 15:
            sid_df["AHI_Severity"] = 1
        elif 15 <= subject_AHI < 30:
            sid_df["AHI_Severity"] = 2
        else:
            sid_df["AHI_Severity"] = 3

        # add BMI target
        subject_BMI = info_df.loc[sid, "BMI"]
        if subject_BMI >= 35:
            sid_df["Obesity"] = 1
        else:
            sid_df["Obesity"] = 0

        sid_df = sid_df.loc[:sid_df[sid_df['Sleep_Stage'].isin(["N1", "N2", "N3", "R", "W"])].last_valid_index(), :]

        all_subjects_fe_df = pd.concat([all_subjects_fe_df, sid_df], ignore_index=True)
    return all_subjects_fe_df, good_quality_sids


def clean_features(all_subjects_fe_df, info_df, nan_feature_names, label_names):
    """
    Cleans the feature dataframe by updating feature names, mapping sleep stages,
    replacing infinite values with NaN, deleting features with excessive missing values,
    and merging additional demographic information. It prepares the data for further 
    analysis by filtering out unnecessary columns and rows with missing values, and 
    returns a cleaned dataframe along with a list of the names of the features that 
    were retained.

    Parameters:
    ----------
    all_subjects_fe_df : pandas dataFrame
        A DataFrame containing the processed features for all subjects meeting the 
        quality threshold.
    info_df : pandas.DataFrame
        A DataFrame containing demographic and clinical information for the subjects, 
        indexed by SID.
    nan_feature_names : list of str
        Names of features that should be considered as NaN and excluded from the analysis.
    label_names : list of str
        Names of columns in the data that are considered as labels and should not be 
        treated as features.

    Returns:
    -------
    clean_df : pandas DataFrame
        The cleaned dataframe after applying all preprocessing steps.
    new_features : list
        A list of the names of the features that were retained in the cleaned DataFrame.
    """
    # update features
    updated_feature_names = [
        f
        for f in all_subjects_fe_df.columns.tolist()
        if f not in nan_feature_names + label_names + ["sid"]
    ]

    # get feature dataframe
    df = all_subjects_fe_df.loc[
        :,
        updated_feature_names + label_names + ["sid"],
    ]

    df.Sleep_Stage = df.Sleep_Stage.map(
        {
            "N1": "N",
            "N2": "N",
            "W": "W",
            "N3": "N",
            "P": "P",
            "R": "R",
            "Missing": "Missing",
        }
    )
    # replace inf
    df = df.replace([np.inf, -np.inf], np.nan)

    # delete features if contains too many nan values
    na_count_df = df.isna().sum()
    features_to_delete = na_count_df[na_count_df > 2000].index.to_list()
    cleaned_feature_names = [
        f for f in updated_feature_names if f not in features_to_delete
    ]

    # select feature columns
    df = df.loc[:, cleaned_feature_names + label_names + ["sid"]]
    # drop columns with nan
    df = df.dropna(how="any", axis=0)
    # add BMI information
    df = pd.merge(df, info_df.loc[:, ["BMI"]], left_on="sid", right_index=True)

    map_stage_to_num = {"P": 1, "N": 0, "R": 0, "W": 1, "Missing": np.nan}
    df["Sleep_Stage"] = df["Sleep_Stage"].map(map_stage_to_num)
    clean_df = df.dropna()

    new_features = clean_df.columns.to_list()
    new_features.remove("sid")
    new_features.remove("Sleep_Stage")
    new_features.remove("Central_Apnea")
    new_features.remove("Obstructive_Apnea")
    new_features.remove("Multiple_Events")
    new_features.remove("Hypopnea")
    new_features.remove("AHI_Severity")
    new_features.remove("Obesity")
    new_features.remove("BMI")
    new_features.remove("circadian_decay")
    new_features.remove("circadian_linear")
    new_features.remove("circadian_cosine")
    new_features.remove("timestamp_start")

    return clean_df, new_features


def data_preparation(threshold, quality_df_dir, features_dir, info_dir):
    """
    Prepare the data for modeling by using data preparation functions

    Parameters:
    ----------
    threshold : float
        The threshold for the percentage of data excluded based on quality scores. 
        Subjects with quality scores below this threshold are considered for analysis.
    
    Returns:
    -------
    clean_df : pandas DataFrame
        The cleaned dataframe after applying all preprocessing steps.
    new_features : list
        A list of the names of the features that were retained in the cleaned DataFrame.
    good_quality_sids : list of str
        A list of subject IDs that met the quality score threshold and were included 
        in the analysis.
    """
    nan_feature_names = [
        "HRV_LF",
        "HRV_LFHF",
        "HRV_LFn",
        "HRV_MSEn",
        "HRV_CMSEn",
        "HRV_RCMSEn",
        "LF_frequency_power",
        "LF_normalized_power",
    ]

    circadian_features = [
        "circadian_decay",
        "circadian_linear",
        "circadian_cosine",
        "timestamp_start",
    ]

    label_names = [
        "Sleep_Stage",
        "Obstructive_Apnea",
        "Central_Apnea",
        "Hypopnea",
        "Multiple_Events",
        "artifact",
    ]
    info_df = pd.read_csv(info_dir)
    all_subjects_fe_df, good_quality_sids = load_data_to_df(
        threshold, quality_df_dir, info_df, features_dir, nan_feature_names, 
        label_names, circadian_features
)
    clean_df, new_features = clean_features(
        all_subjects_fe_df, info_df, nan_feature_names, label_names
    )
    return clean_df, new_features, good_quality_sids


def split_data(new_df, good_quality_sids, features):
    """
    Splits the dataset into a subset with reduced feature set by removing 
    highly correlated features.

    Parameters:
    ----------
    new_df : pandas DataFrame
        The dataframe containing features and labels for all subjects.
    good_quality_sids : list of str
        A list of subject IDs that met the quality score threshold and were 
        included in the analysis.
    features : list
        A list of feature names to consider for correlation analysis and 
        potential removal.

    Returns:
    -------
    SW_df : pandas DataFrame
        The dataframe with reduced features based on correlation analysis.
    final_features : list
        The list of features retained after removing highly correlated ones.
    """
    train_sids = good_quality_sids[:45]

    corr_matrix = new_df.loc[new_df["sid"].isin(train_sids), features].corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))

    # Find features with correlation greater than a threshold (e.g., 0.8 or 0.9)
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    if "ACC_INDEX" in to_drop:
        to_drop.remove("ACC_INDEX")

    # Drop highly correlated features
    df_reduced = new_df.drop(to_drop, axis=1)

    final_features = [f for f in features if f in df_reduced.columns]

    SW_df = df_reduced.copy()

    return SW_df, final_features


def missingness_imputation(data):
    """Perform missingness imputation on the given data.

    Parameters
    ----------
    data : array-like
        The data to be imputed.

    Returns
    -------
    interpolated_series : pandas Series
        The imputed data.
    """

    indices = np.arange(len(data))
    series = pd.Series(data, index=indices)
    interpolated_series = series.interpolate(method="linear")
    return interpolated_series


def half_gaussian_kernel(size, std_dev):
    """Create a half Gaussian kernel.

    Parameters
    ----------
    size : int
        The size of the kernel.
    std_dev : float
        The standard deviation of the kernel.

    Returns
    -------
    half_kernel : array
        The half Gaussian kernel.
    """
    full_kernel = gaussian(size, std_dev)
    half_kernel = full_kernel[: size // 2]
    half_kernel /= half_kernel.sum()
    return half_kernel


def apply_half_gaussian_filter(data, kernel_size, std_dev):
    """Apply a half Gaussian filter to the given data.

    Parameters
    ----------
    data : array-like
        The data to be filtered.
    kernel_size : int
        The size of the kernel.
    std_dev : float
        The standard deviation of the kernel.

    Returns
    -------
    filtered_data : array
        The filtered data.
    """
    kernel = half_gaussian_kernel(kernel_size, std_dev)
    filtered_data = convolve(data, kernel, mode="valid")
    left_padding_length = kernel_size // 2 - 1
    filtered_data = np.pad(
        filtered_data, (left_padding_length, 0), "constant", constant_values=(np.nan,)
    )
    return filtered_data


def half_gaussian_filtering(df, columns, kernel_size=40, std_dev=100):
    """Perform half Gaussian filtering on the given DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame to be filtered.
    columns : list
        The columns to be filtered.
    kernel_size : int
        The size of the kernel.
    std_dev : float
        The standard deviation of the kernel.

    Returns
    -------
    df : pandas DataFrame
        The filtered DataFrame.
    """
    for column in columns:
        interpolated_series = missingness_imputation(
            apply_half_gaussian_filter(df[column], kernel_size, std_dev)
        )
        df["gaussian_{}".format(column)] = interpolated_series
    return df


def apply_gaussian_filter(data, kernel_size, std_dev):
    """Apply a Gaussian filter to the given data.

    Parameters
    ----------
    data : array-like
        The data to be filtered.
    kernel_size : int
        The size of the kernel.
    std_dev : float
        The standard deviation of the kernel.

    Returns
    -------
    filtered_data : array
        The filtered data.
    """
    kernel = windows.gaussian(kernel_size, std_dev, sym=True)
    kernel /= np.sum(kernel)
    filtered_data = convolve(data, kernel, mode="same")
    return filtered_data


def gaussian_filtering(df, columns, kernel_size=40, std_dev=100):
    """Perform Gaussian filtering on the given DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame to be filtered.
    columns : list
        The columns to be filtered.
    kernel_size : int
        The size of the kernel.
    std_dev : float
        The standard deviation of the kernel.

    Returns
    -------
    df : pandas DataFrame
        The filtered DataFrame.
    """
    for column in columns:
        interpolated_series = missingness_imputation(
            apply_gaussian_filter(df[column], kernel_size, std_dev)
        )
        df["gaussian_{}".format(column)] = interpolated_series
    #         df["gaussian_diff_{}".format(column)] = interpolated_series - df[column]
    return df


def rolling_stds(df, columns, window_size=20):
    """Calculate rolling standard deviations for the given columns in the DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame for which to calculate rolling standard deviations.
    columns : list
        The columns for which to calculate rolling standard deviations.
    window_size : int
        The size of the rolling window.

    Returns
    -------
    df : pandas DataFrame
        The DataFrame with the rolling standard deviations added.
    """
    for column in columns:
        df["rolling_var_{}".format(column)] = (
            df[column].rolling(window=window_size, min_periods=1).var()
        )
    return df



def add_derivatives(df, features):
    """Add first and second derivatives to the given features in the DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame to which to add the derivatives.
    features : list
        The features to which to add the derivatives.
    

    Returns
    -------
    df : pandas DataFrame
        The DataFrame with the derivatives added.
    """
    for feature in features:
        # First derivative
        first_derivative_column = "gaussian_" + feature + "_1st_derivative"
        df[first_derivative_column] = np.gradient(df["gaussian_" + feature])

        # Second derivative
        raw_derivative_column = "raw_" + feature + "_1st_derivative"
        df[raw_derivative_column] = df[feature].diff()
    return df


def get_variable(group_variables, idx):
    """
    Pick variable(s) from the given list based on the provided index.

    Parameters:
    ----------
    group_variables : list
        A list of variables from which to select.
    idx : int
        The index specifying which variable(s) to retrieve. Expected values are 0, 1, or 2.
        An index of 0 or 1 returns a list with the respective single variable, while an 
        ndex of 2 returns the entire list.

    Returns:
    ----------
    group_variable: list
        A list containing the selected variable(s).
    """
    if idx == 0:
        group_variable = [group_variables[idx]]
    elif idx == 1:
        group_variable = [group_variables[idx]]
    elif idx == 2:
        group_variable = group_variables
    else:
        print('Wrong index')
    return group_variable


def calculate_qaulity_score(feature_df_dir):
    """ Code used to calculate quality score of each participant
    """
    files = os.listdir(feature_df_dir)
    files.sort()

    sids = []
    total_segments = []
    num_excludes = []
    percentages = []

    for file in files:
        sid = file.split('_')[0]
        file_path = str(feature_df_dir + file)
        df = pd.read_csv(file_path)
        segment_len = len(df)
        segment_exclude = np.sum(df.artifact)
        percentage = segment_exclude / segment_len

        sids.append(sid)
        total_segments.append(segment_len)
        num_excludes.append(segment_exclude)
        percentages.append(percentage)

    qs = pd.DataFrame({'sid': sids,
                    'total_segments': total_segments,
                    'num_excludes': num_excludes,
                    'percentage_excludes': percentages})
    qs.to_csv('quality_scores_per_subject.csv', index=False)



