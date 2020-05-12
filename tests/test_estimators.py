import pytest
import numpy.testing as npt

import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None

from rigid_body_motion.estimators import shortest_arc_rotation


class TestEstimators(object):
    def test_shortest_arc_rotation(self):
        """"""
        # ndarray
        v1 = np.zeros((10, 3))
        v1[:, 0] = 1.0
        v2 = np.zeros((10, 3))
        v2[:, 1] = 1.0
        q_exp = np.tile((np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2), (10, 1))

        npt.assert_allclose(shortest_arc_rotation(v1, v2), q_exp)

    @pytest.mark.skipif(xr is None, reason="xarray not installed")
    def test_shortest_arc_rotation_xr(self):
        """"""
        v1 = np.zeros((10, 3))
        v1[:, 0] = 1.0
        v2 = np.zeros((10, 3))
        v2[:, 1] = 1.0
        q_exp = np.tile((np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2), (10, 1))

        v1_da = xr.DataArray(
            v1, {"cartesian_axis": ["x", "y", "z"]}, ("time", "cartesian_axis")
        )
        expected = xr.DataArray(
            q_exp,
            {"quaternion_axis": ["w", "x", "y", "z"]},
            ("time", "quaternion_axis"),
        )
        actual = shortest_arc_rotation(v1_da, v2, dim="cartesian_axis")

        xr.testing.assert_allclose(actual, expected)
