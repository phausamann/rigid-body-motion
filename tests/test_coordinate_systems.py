import numpy as np
import pytest
from numpy import testing as npt

import rigid_body_motion as rbm
from rigid_body_motion.coordinate_systems import _replace_dim


class TestCoordinateSystems(object):
    def test_replace_dim(self):
        """"""
        xr = pytest.importorskip("xarray")

        da = xr.DataArray(
            np.ones((10, 3)),
            {
                "time": np.arange(10),
                "old_dim": ["a", "b", "c"],
                "old_dim_coord": ("old_dim", np.arange(3)),
            },
            ("time", "old_dim"),
        )
        dims = da.dims
        coords = dict(da.coords)

        new_coords, new_dims = _replace_dim(coords, dims, -1, "cartesian", 2)
        assert set(new_coords.keys()) == set(new_dims)
        assert new_coords["cartesian_axis"] == ["x", "y"]
        assert new_dims == ("time", "cartesian_axis")

        new_coords, new_dims = _replace_dim(coords, dims, -1, "polar", 2)
        assert set(new_coords.keys()) == set(new_dims)
        assert new_coords["polar_axis"] == ["r", "phi"]
        assert new_dims == ("time", "polar_axis")

        new_coords, new_dims = _replace_dim(coords, dims, -1, "cartesian", 3)
        assert set(new_coords.keys()) == set(new_dims)
        assert new_coords["cartesian_axis"] == ["x", "y", "z"]
        assert new_dims == ("time", "cartesian_axis")

        new_coords, new_dims = _replace_dim(coords, dims, -1, "spherical", 3)
        assert set(new_coords.keys()) == set(new_dims)
        assert new_coords["spherical_axis"] == ["r", "theta", "phi"]
        assert new_dims == ("time", "spherical_axis")

        new_coords, new_dims = _replace_dim(coords, dims, -1, "quaternion", 3)
        assert set(new_coords.keys()) == set(new_dims)
        assert new_coords["quaternion_axis"] == ["w", "x", "y", "z"]
        assert new_dims == ("time", "quaternion_axis")

    def test_cartesian_to_polar(self):
        """"""
        arr = np.ones((10, 2))
        expected = np.tile((np.sqrt(2), np.pi / 4), (10, 1))
        actual = rbm.cartesian_to_polar(arr, axis=1)
        npt.assert_almost_equal(actual, expected)

        with pytest.raises(ValueError):
            rbm.cartesian_to_polar(np.ones((10, 3)), axis=1)

    def test_polar_to_cartesian(self):
        """"""
        arr = np.tile((np.sqrt(2), np.pi / 4), (10, 1))
        expected = np.ones((10, 2))
        actual = rbm.polar_to_cartesian(arr, axis=1)
        npt.assert_almost_equal(actual, expected)

        with pytest.raises(ValueError):
            rbm.polar_to_cartesian(np.ones((10, 3)), axis=1)

    def test_cartesian_to_spherical(self):
        """"""
        arr = np.ones((10, 3))
        expected = np.tile((np.sqrt(3), 0.9553166, np.pi / 4), (10, 1))
        actual = rbm.cartesian_to_spherical(arr, axis=1)
        npt.assert_almost_equal(actual, expected)

        with pytest.raises(ValueError):
            rbm.cartesian_to_spherical(np.ones((10, 2)), axis=1)

    def test_spherical_to_cartesian(self):
        """"""
        arr = np.tile((np.sqrt(3), 0.9553166, np.pi / 4), (10, 1))
        expected = np.ones((10, 3))
        actual = rbm.spherical_to_cartesian(arr, axis=1)
        npt.assert_almost_equal(actual, expected)

        with pytest.raises(ValueError):
            rbm.spherical_to_cartesian(np.ones((10, 2)), axis=1)
