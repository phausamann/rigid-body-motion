""""""
import numpy as np
from quaternion import as_float_array, from_rotation_vector


def make_test_motion(
    n_samples, freq=1, max_angle=np.pi / 2, fs=100, stack=True, inverse=False,
):
    """ Create sinusoidal linear and angular motion around all three axes. """
    import pandas as pd

    if stack:
        n_samples = n_samples // 3

    if inverse:
        max_angle = -max_angle

    trajectory = max_angle * np.sin(
        2 * np.pi * freq * np.arange(n_samples) / fs
    )

    tx = trajectory[:, None] * np.array((1.0, 0.0, 0.0))[None, :]
    ty = trajectory[:, None] * np.array((0.0, 1.0, 0.0))[None, :]
    tz = trajectory[:, None] * np.array((0.0, 0.0, 1.0))[None, :]

    if stack:
        translation = np.vstack((tx, ty, tz))
    else:
        translation = tx + ty + tz

    rotation = as_float_array(from_rotation_vector(translation))

    timestamps = pd.date_range(start=0, periods=n_samples, freq=f"{1/fs}S")

    return translation, rotation, timestamps
