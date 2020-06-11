#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import sys
from loguru import logger
from pprint import pformat
from typing import List

import numpy as np
import pandas as pd
import scipy
import six
import sklearn.preprocessing as pre
import torch
import tqdm
import yaml

import augment
import dataset

# Some defaults for non-specified arguments in yaml
DEFAULT_ARGS = {
    'outputpath': 'experiments',
    'loss': 'BCELoss',
    'batch_size': 64,
    'num_workers': 4,
    'epochs': 100,
    'transforms': [],
    'label_type':'soft',
    'scheduler_args': {
        'patience': 3,
        'factor': 0.1,
    },
    'early_stop': 7,
    'optimizer': 'Adam',
    'optimizer_args': {
        'lr': 0.001,
    },
    'threshold': None,  #Default threshold for postprocessing function
    'postprocessing': 'double',
}


def parse_config_or_kwargs(config_file, **kwargs):
    """parse_config_or_kwargs

    :param config_file: Config file that has parameters, yaml format
    :param **kwargs: Other alternative parameters or overwrites for config
    """
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
    # values from config file are all possible params
    arguments = dict(yaml_config, **kwargs)
    # In case some arguments were not passed, replace with default ones
    for key, value in DEFAULT_ARGS.items():
        arguments.setdefault(key, value)
    return arguments


def find_contiguous_regions(activity_array):
    """Find contiguous regions from bool valued numpy.array.
    Copy of https://dcase-repo.github.io/dcase_util/_modules/dcase_util/data/decisions.html#DecisionEncoder

    Reason is:
    1. This does not belong to a class necessarily
    2. Import DecisionEncoder requires sndfile over some other imports..which causes some problems on clusters

    """

    # Find the changes in the activity_array
    change_indices = np.logical_xor(activity_array[1:],
                                    activity_array[:-1]).nonzero()[0]

    # Shift change_index with one, focus on frame after the change.
    change_indices += 1

    if activity_array[0]:
        # If the first element of activity_array is True add 0 at the beginning
        change_indices = np.r_[0, change_indices]

    if activity_array[-1]:
        # If the last element of activity_array is True, add the length of the array
        change_indices = np.r_[change_indices, activity_array.size]

    # Reshape the result into two columns
    return change_indices.reshape((-1, 2))


def split_train_cv(input_data, frac: float = 0.9, **kwargs):
    """split_train_cv

    :param data_frame:
    :param frac:
    :type frac: float
    """
    if isinstance(input_data, list):
        N = len(input_data)
        indicies = np.random.permutation(N)
        train_size = round(N * frac)
        cv_size = N - train_size
        train_idxs, cv_idxs = indicies[:train_size], indicies[cv_size:]
        input_data = np.array(input_data)
        return input_data[train_idxs].tolist(), input_data[cv_idxs].tolist()
    elif isinstance(input_data, pd.DataFrame):
        train_df = input_data.sample(frac=frac)
        cv_df = input_data[~input_data.index.isin(train_df.index)]
        return train_df, cv_df


def parse_transforms(transform_list):
    """parse_transforms
    parses the config files transformation strings to coresponding methods

    :param transform_list: String list
    """
    transforms = []
    for trans in transform_list:
        if trans == 'noise':
            transforms.append(augment.GaussianNoise(snr=25))
        elif trans == 'roll':
            transforms.append(augment.Roll(0, 10))
        elif trans == 'freqmask':
            transforms.append(augment.FreqMask(2, 8))
        elif trans == 'timemask':
            transforms.append(augment.TimeMask(2, 60))
        elif trans == 'crop':
            transforms.append(augment.RandomCrop(200))
        elif trans == 'randompad':
            transforms.append(augment.RandomPad(value=0., padding=25))
        elif trans == 'flipsign':
            transforms.append(augment.FlipSign())
        elif trans == 'shift':
            transforms.append(augment.Shift())
    return torch.nn.Sequential(*transforms)


def pprint_dict(in_dict, outputfun=sys.stdout.write, formatter='yaml'):
    """pprint_dict

    :param outputfun: function to use, defaults to sys.stdout
    :param in_dict: dict to print
    """
    if formatter == 'yaml':
        format_fun = yaml.dump
    elif formatter == 'pretty':
        format_fun = pformat
    for line in format_fun(in_dict).split('\n'):
        outputfun(line)


def getfile_outlogger(outputfile):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    if outputfile:
        logger.add(outputfile, enqueue=True, format=log_format)
    return logger


def train_labelencoder(labels: pd.Series, sparse=True):
    """encode_labels

    Encodes labels

    :param labels: pd.Series representing the raw labels e.g., Speech, Water
    :param encoder (optional): Encoder already fitted 
    returns encoded labels (many hot) and the encoder
    """
    assert isinstance(labels, pd.Series), "Labels need to be series"
    if isinstance(labels[0], six.string_types):
        # In case of using non processed strings, e.g., Vaccum, Speech
        label_array = labels.str.split(',').values.tolist()
    elif isinstance(labels[0], np.ndarray):
        # Encoder does not like to see numpy array
        label_array = [lab.tolist() for lab in labels]
    elif isinstance(labels[0], collections.Iterable):
        label_array = labels
    encoder = pre.MultiLabelBinarizer(sparse_output=sparse)
    encoder.fit(label_array)
    return encoder


def encode_labels(labels: pd.Series, encoder=None, sparse=True):
    """encode_labels

    Encodes labels

    :param labels: pd.Series representing the raw labels e.g., Speech, Water
    :param encoder (optional): Encoder already fitted 
    returns encoded labels (many hot) and the encoder
    """
    assert isinstance(labels, pd.Series), "Labels need to be series"
    instance = labels.iloc[0]
    if isinstance(instance, six.string_types):
        # In case of using non processed strings, e.g., Vaccum, Speech
        label_array = labels.str.split(',').values.tolist()
    elif isinstance(instance, np.ndarray):
        # Encoder does not like to see numpy array
        label_array = [lab.tolist() for lab in labels]
    elif isinstance(instance, collections.Iterable):
        label_array = labels
    if not encoder:
        encoder = pre.MultiLabelBinarizer(sparse_output=sparse)
        encoder.fit(label_array)
    labels_encoded = encoder.transform(label_array)
    return labels_encoded, encoder

    # return pd.arrays.SparseArray(
    # [row.toarray().ravel() for row in labels_encoded]), encoder


def decode_with_timestamps(encoder: pre.MultiLabelBinarizer, labels: np.array):
    """decode_with_timestamps
    Decodes the predicted label array (2d) into a list of
    [(Labelname, onset, offset), ...]

    :param encoder: Encoder during training
    :type encoder: pre.MultiLabelBinarizer
    :param labels: n-dim array
    :type labels: np.array
    """
    if labels.ndim == 3:
        return [_decode_with_timestamps(encoder, lab) for lab in labels]
    else:
        return _decode_with_timestamps(encoder, labels)


def sma_filter(x, window_size, axis=1):
    """sma_filter

    :param x: Input numpy array,
    :param window_size: filter size
    :param axis: over which axis ( usually time ) to apply
    """
    # 1 is time axis
    kernel = np.ones((window_size, )) / window_size

    def moving_average(arr):
        return np.convolve(arr, kernel, 'same')

    return np.apply_along_axis(moving_average, axis, x)


def median_filter(x, window_size, threshold=0.5):
    """median_filter

    :param x: input prediction array of shape (B, T, C) or (B, T).
        Input is a sequence of probabilities 0 <= x <= 1
    :param window_size: An integer to use 
    :param threshold: Binary thresholding threshold
    """
    x = binarize(x, threshold=threshold)
    if x.ndim == 3:
        size = (1, window_size, 1)
    elif x.ndim == 2 and x.shape[0] == 1:
        # Assume input is class-specific median filtering
        # E.g, Batch x Time  [1, 501]
        size = (1, window_size)
    elif x.ndim == 2 and x.shape[0] > 1:
        # Assume input is standard median pooling, class-independent
        # E.g., Time x Class [501, 10]
        size = (window_size, 1)
    return scipy.ndimage.median_filter(x, size=size)


def _decode_with_timestamps(encoder, labels):
    result_labels = []
    for i, label_column in enumerate(labels.T):
        change_indices = find_contiguous_regions(label_column)
        # append [onset, offset] in the result list
        for row in change_indices:
            result_labels.append((encoder.classes_[i], row[0], row[1]))
    return result_labels


def inverse_transform_labels(encoder, pred):
    if pred.ndim == 3:
        return [encoder.inverse_transform(x) for x in pred]
    else:
        return encoder.inverse_transform(pred)


def binarize(pred, threshold=0.5):
    # Batch_wise
    if pred.ndim == 3:
        return np.array(
            [pre.binarize(sub, threshold=threshold) for sub in pred])
    else:
        return pre.binarize(pred, threshold=threshold)


def double_threshold(x, high_thres, low_thres, n_connect=1):
    """double_threshold
    Helper function to calculate double threshold for n-dim arrays

    :param x: input array
    :param high_thres: high threshold value
    :param low_thres: Low threshold value
    :param n_connect: Distance of <= n clusters will be merged
    """
    assert x.ndim <= 3, "Whoops something went wrong with the input ({}), check if its <= 3 dims".format(
        x.shape)
    if x.ndim == 3:
        apply_dim = 1
    elif x.ndim < 3:
        apply_dim = 0
    # x is assumed to be 3d: (batch, time, dim)
    # Assumed to be 2d : (time, dim)
    # Assumed to be 1d : (time)
    # time axis is therefore at 1 for 3d and 0 for 2d (
    return np.apply_along_axis(lambda x: _double_threshold(
        x, high_thres, low_thres, n_connect=n_connect),
                               axis=apply_dim,
                               arr=x)


def _double_threshold(x, high_thres, low_thres, n_connect=1, return_arr=True):
    """_double_threshold
    Computes a double threshold over the input array

    :param x: input array, needs to be 1d
    :param high_thres: High threshold over the array
    :param low_thres: Low threshold over the array
    :param n_connect: Postprocessing, maximal distance between clusters to connect
    :param return_arr: By default this function returns the filtered indiced, but if return_arr = True it returns an array of tsame size as x filled with ones and zeros.
    """
    assert x.ndim == 1, "Input needs to be 1d"
    high_locations = np.where(x > high_thres)[0]
    locations = x > low_thres
    encoded_pairs = find_contiguous_regions(locations)

    filtered_list = list(
        filter(
            lambda pair:
            ((pair[0] <= high_locations) & (high_locations <= pair[1])).any(),
            encoded_pairs))

    filtered_list = connect_(filtered_list, n_connect)
    if return_arr:
        zero_one_arr = np.zeros_like(x, dtype=int)
        for sl in filtered_list:
            zero_one_arr[sl[0]:sl[1]] = 1
        return zero_one_arr
    return filtered_list


def connect_clusters(x, n=1):
    if x.ndim == 1:
        return connect_clusters_(x, n)
    if x.ndim >= 2:
        return np.apply_along_axis(lambda a: connect_clusters_(a, n=n), -2, x)


def connect_clusters_(x, n=1):
    """connect_clusters_
    Connects clustered predictions (0,1) in x with range n

    :param x: Input array. zero-one format
    :param n: Number of frames to skip until connection can be made
    """
    assert x.ndim == 1, "input needs to be 1d"
    reg = find_contiguous_regions(x)
    start_end = connect_(reg, n=n)
    zero_one_arr = np.zeros_like(x, dtype=int)
    for sl in start_end:
        zero_one_arr[sl[0]:sl[1]] = 1
    return zero_one_arr


def connect_(pairs, n=1):
    """connect_
    Connects two adjacent clusters if their distance is <= n

    :param pairs: Clusters of iterateables e.g., [(1,5),(7,10)]
    :param n: distance between two clusters 
    """
    if len(pairs) == 0:
        return []
    start_, end_ = pairs[0]
    new_pairs = []
    for i, (next_item, cur_item) in enumerate(zip(pairs[1:], pairs[0:])):
        end_ = next_item[1]
        if next_item[0] - cur_item[1] <= n:
            pass
        else:
            new_pairs.append((start_, cur_item[1]))
            start_ = next_item[0]
    new_pairs.append((start_, end_))
    return new_pairs


def predictions_to_time(df, ratio):
    df.onset = df.onset * ratio
    df.offset = df.offset * ratio
    return df


def estimate_scaler(dataloader, **scaler_args):

    scaler = pre.StandardScaler(**scaler_args)
    with tqdm.tqdm(total=len(dataloader),
                   unit='batch',
                   leave=False,
                   desc='Estimating Scaler') as pbar:
        for batch in dataloader:
            feature = batch[0]
            # Flatten time and batch dim to one
            feature = feature.reshape(-1, feature.shape[-1])
            pbar.set_postfix(feature=feature.shape)
            pbar.update()
            scaler.partial_fit(feature)
    return scaler


def rescale_0_1(x):
    if x.ndim == 2:
        return pre.minmax_scale(x, axis=0)
    else:

        def min_max_scale(a):
            return pre.minmax_scale(a, axis=0)

def df_to_dict(df, index='filename', value='hdf5path'):
    return dict(zip(df[index],df[value]))
