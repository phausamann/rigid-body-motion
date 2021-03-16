""""""
import numpy as np
from quaternion import as_float_array, as_quat_array, from_rotation_vector

from rigid_body_motion.utils import rotate_vectors


def make_test_motion(
    n_samples,
    freq=1,
    max_angle=np.pi / 2,
    fs=1000,
    stack=True,
    inverse=False,
    offset=None,
):
    """ Create sinusoidal linear and angular motion around all three axes. """
    import pandas as pd

    if inverse:
        max_angle = -max_angle

    trajectory = (
        max_angle
        * np.sin(2 * np.pi * freq * np.arange(n_samples) / fs)[:, np.newaxis]
    )

    if stack:
        ax = np.array((1.0, 0.0, 0.0))[np.newaxis, :]
        ay = np.array((0.0, 1.0, 0.0))[np.newaxis, :]
        az = np.array((0.0, 0.0, 1.0))[np.newaxis, :]
        tx, ty, tz = np.array_split(trajectory, 3)
        translation = np.vstack((tx * ax, ty * ay, tz * az))
    else:
        translation = np.tile(trajectory, (1, 3))

    rotation = as_float_array(from_rotation_vector(translation))

    if offset is not None:
        translation += rotate_vectors(
            as_quat_array(rotation), np.array(offset)[np.newaxis, :]
        )

    timestamps = pd.date_range(start=0, periods=n_samples, freq=f"{1/fs}S")

    return translation, rotation, timestamps
