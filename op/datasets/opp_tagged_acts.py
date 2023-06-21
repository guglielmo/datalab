# -*- coding: utf-8 -*-
"""Openparlamento tagged acts dataset for topics classification.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter

from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import _remove_long_seq
import pandas as pd
import numpy as np
import json
import warnings


def load_data_16(path='tagged_acts_16.npz', test_split=0.2, seed=113, **kwargs):
    """Loads the Openparlamento tagged acts classification dataset for Leg 16

    # Arguments
        path: where to cache the data (relative to `~/.keras/dataset`).
        test_split: Fraction of the dataset to be used as test data.
        seed: random seed for sample shuffling.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    Note that raw data are returned, that are going to need further pre-processing
    in order to work in deep learning models.
    """
    path = get_file(path,
                    origin='https://opp-datasets.s3.eu-central-1.amazonaws.com/tagged_acts_16.npz',
                    file_hash='4c3556727beb2c4cdd65813327d42b16')
    with np.load(path, allow_pickle=True) as f:
        texts, labels = f['texts'], f['labels']

    rng = np.random.RandomState(seed)
    indices = np.arange(len(texts))
    rng.shuffle(indices)
    texts = texts[indices]
    labels = labels[indices]

    idx = int(len(texts) * (1 - test_split))
    x_train, y_train = np.array(texts[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(texts[idx:]), np.array(labels[idx:])

    return (x_train, y_train), (x_test, y_test)


def load_vocab_16(path='tagged_acts_16_vocab.json'):
    """Load the vocabulary (as a Counter, with counts of instance) for Leg 16

    # Arguments
        path: where to cache the data (relative to `~/.keras/dataset`).

    # Returns
        The word index dictionary.
    """
    path = get_file(
        path,
        origin='https://opp-datasets.s3.eu-central-1.amazonaws.com/tagged_acts_16_vocab.json',
        file_hash='02826d2948bd04bddfb68e56f44b8ca8')
    with open(path) as f:
        return Counter(json.load(f))


