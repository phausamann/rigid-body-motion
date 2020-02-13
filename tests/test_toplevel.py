import pytest
import numpy.testing as npt
from .helpers import mock_quaternion, register_rf_tree

import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None

import rigid_body_motion as rbm


class TestTopLevel(object):
    """"""

    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """"""
        rbm.clear_registry()
        yield

    @pytest.fixture()
    def rf_tree(self):
        """"""
        register_rf_tree(tc1=(1., 0., 0.), tc2=(-1., 0., 0.),
                         rc2=mock_quaternion(np.pi, 0., 0.))

    def test_transform_points(self, rf_tree):
        """"""
        arr_child2 = (1., 1., 1.)
        arr_exp = (-3., -1., 1.)

        # tuple
        arr_child1 = rbm.transform_points(
            arr_child2, outof='child2', into='child1')
        npt.assert_almost_equal(arr_child1, arr_exp)

    @pytest.mark.skipif(xr is None, reason='xarray not installed')
    def test_transform_points_xr(self, rf_tree):
        """"""
        arr_child2 = (1., 1., 1.)
        arr_exp = (-3., -1., 1.)

        da_child2 = xr.DataArray(
            np.tile(arr_child2, (10, 1)), {'time': np.arange(10)},
            ('time', 'cartesian_axis'))
        da_child1 = rbm.transform_points(
            da_child2, outof='child2', into='child1', dim='cartesian_axis',
            timestamps='time')
        assert da_child1.shape == (10, 3)
        npt.assert_almost_equal(da_child1[0], arr_exp)

    def test_transform_quaternions(self, rf_tree):
        """"""
        arr_child2 = (1., 0., 0., 0.)
        arr_exp = mock_quaternion(np.pi, 0., 0.)

        # tuple
        arr_child1 = rbm.transform_quaternions(
            arr_child2, outof='child2', into='child1')
        npt.assert_almost_equal(arr_child1, arr_exp)

    @pytest.mark.skipif(xr is None, reason='xarray not installed')
    def test_transform_quaternions_xr(self, rf_tree):
        """"""
        arr_child2 = (1., 0., 0., 0.)
        arr_exp = mock_quaternion(np.pi, 0., 0.)

        da_child2 = xr.DataArray(
            np.tile(arr_child2, (10, 1)), {'time': np.arange(10)},
            ('time', 'quaternion_axis'))
        da_child1 = rbm.transform_quaternions(
            da_child2, outof='child2', into='child1', dim='quaternion_axis',
            timestamps='time')
        assert da_child1.shape == (10, 4)
        npt.assert_almost_equal(da_child1[0], arr_exp)

    def test_transform_vectors(self, rf_tree):
        """"""
        arr_child2 = (1., 1., 1.)
        arr_exp = (-1., -1., 1.)

        # tuple
        arr_child1 = rbm.transform_vectors(
            arr_child2, outof='child2', into='child1')
        npt.assert_almost_equal(arr_child1, arr_exp)

    @pytest.mark.skipif(xr is None, reason='xarray not installed')
    def test_transform_vectors_xr(self, rf_tree):
        """"""
        arr_child2 = (1., 1., 1.)
        arr_exp = (-1., -1., 1.)

        da_child2 = xr.DataArray(
            np.tile(arr_child2, (10, 1)), {'time': np.arange(10)},
            ('time', 'cartesian_axis'))
        da_child1 = rbm.transform_vectors(
            da_child2, outof='child2', into='child1', dim='cartesian_axis',
            timestamps='time')
        assert da_child1.shape == (10, 3)
        npt.assert_almost_equal(da_child1[0], arr_exp)

    def test_transform_coordinates(self):
        """"""
        # ndarray
        arr = np.ones((10, 2))
        expected = np.tile((np.sqrt(2), np.pi/4), (10, 1))
        actual = rbm.transform_coordinates(
            arr, outof='cartesian', into='polar', axis=1)
        npt.assert_equal(actual, expected)

        with pytest.raises(ValueError):
            rbm.transform_coordinates(
                np.ones((10, 3)), outof='cartesian', into='polar')
        with pytest.raises(ValueError):
            rbm.transform_coordinates(
                np.ones((10, 2)), outof='unsupported', into='polar')

    @pytest.mark.skipif(xr is None, reason='xarray not installed')
    def test_transform_coordinates_xr(self):
        """"""
        # DataArray
        da = xr.DataArray(
            np.ones((10, 3)),
            {'time': np.arange(10), 'cartesian_axis': ['x', 'y', 'z']},
            ('time', 'cartesian_axis'))
        expected = xr.DataArray(
            np.tile((1.732051, 0.955317, 0.785398), (10, 1)),
            {'time': np.arange(10), 'spherical_axis': ['r', 'theta', 'phi']},
            ('time', 'spherical_axis'))

        actual = rbm.transform_coordinates(
            da, outof='cartesian', into='spherical')
        xr.testing.assert_allclose(actual, expected)
