# -*- coding: utf-8 -*-

"""Top-level package for Rigid Body Motion."""

__author__ = """Peter Hausamann"""
__email__ = 'peter@hausamann.de'
__version__ = '0.1.0'


import numpy as np


def _2d_cartesian_to_polar(arr, axis):
    """"""
    if arr.shape[axis] != 2:
        raise ValueError(
            'Expected shape along axis {} to be 2, got {} instead.'.format(
                axis, arr.shape[axis]))

    r = np.linalg.norm(arr, axis=axis)
    phi = np.arctan2(np.take(arr, 1, axis=axis), np.take(arr, 0, axis=axis))
    return np.stack((r, phi), axis=axis)


class transform(object):
    """"""

    def __init__(self, arr, axis=-1):
        """"""
        self.arr = arr
        self.axis = axis
        self.src = None

    def from_(self, src):
        """"""
        self.src = src
        return self

    def to_(self, dst):
        """"""
        if self.src is not None:
            if self.src == 'cartesian' and dst == 'polar':
                return _2d_cartesian_to_polar(self.arr, self.axis)
            else:
                raise ValueError(
                    'Unsupported transformation: {} to {}.'.format(
                        self.src, dst))
        else:
            raise ValueError(
                'Unspecified source reference frame or coordinate system.')
