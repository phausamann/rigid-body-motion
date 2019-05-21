# -*- coding: utf-8 -*-

"""Top-level package for Rigid Body Motion."""

__author__ = """Peter Hausamann"""
__email__ = 'peter@hausamann.de'
__version__ = '0.1.0'


import numpy as np


def cartesian_to_polar_2d(arr, axis=-1):
    """"""
    if arr.shape[axis] != 2:
        raise ValueError(
            'Expected shape along axis {} to be 2, got {} instead.'.format(
                axis, arr.shape[axis]))

    r = np.linalg.norm(arr, axis=axis)
    phi = np.arctan2(np.take(arr, 1, axis=axis), np.take(arr, 0, axis=axis))
    return np.stack((r, phi), axis=axis)


def polar_to_cartesian_2d(arr, axis=-1):
    """"""
    if arr.shape[axis] != 2:
        raise ValueError(
            'Expected shape along axis {} to be 2, got {} instead.'.format(
                axis, arr.shape[axis]))

    x = np.take(arr, 0, axis=axis) * np.cos(np.take(arr, 1, axis=axis))
    y = np.take(arr, 0, axis=axis) * np.sin(np.take(arr, 1, axis=axis))
    return np.stack((x, y), axis=axis)


_cs_funcs = {
    'cartesian': {'polar': {2: cartesian_to_polar_2d}}}


def transform(arr, outof=None, into=None, axis=-1):
    """"""
    dim = arr.shape[axis]
    try:
        transform_func = _cs_funcs[outof][into][dim]
    except KeyError:
        raise ValueError(
            'Unsupported transformation: {} to {} in {} dimensions.'.format(
                outof, into, dim))

    return transform_func(arr, axis=axis)
