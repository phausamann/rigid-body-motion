import numpy as np
import pytest
from numpy import testing as npt

from rigid_body_motion.estimators import (
    _reshape_vectors,
    best_fit_transform,
    iterative_closest_point,
    shortest_arc_rotation,
)


class TestEstimators(object):
    def test_reshape_vectors(self):
        """"""
        v = np.random.randn(10, 3, 10)
        vt1, vt2, was_datarray = _reshape_vectors(v, v, 1, None)
        assert vt1.shape == (100, 3)
        assert vt2.shape == (100, 3)
        assert not was_datarray

        # not 3d
        with pytest.raises(ValueError):
            _reshape_vectors(v[:, :2], v[:, :2], 1, None)

        # not enough dimensions
        with pytest.raises(ValueError):
            _reshape_vectors(v[0, :, 0], v[0, :, 0], 0, None)

        # different shapes
        with pytest.raises(ValueError):
            _reshape_vectors(v, v[1:], 1, None)

    def test_shortest_arc_rotation(self):
        """"""
        # ndarray
        v1 = np.zeros((10, 3))
        v1[:, 0] = 1.0
        v2 = np.zeros((10, 3))
        v2[:, 1] = 1.0
        q_exp = np.tile((np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2), (10, 1))

        npt.assert_allclose(shortest_arc_rotation(v1, v2), q_exp)

    def test_shortest_arc_rotation_xr(self):
        """"""
        xr = pytest.importorskip("xarray")

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

    def test_best_fit_transform(self, get_rf_tree, mock_quaternion):
        """"""
        rf_world, rf_child1, _ = get_rf_tree(
            (1.0, 0.0, 0.0), mock_quaternion(np.pi / 2, 0.0, 0.0)
        )
        v1 = np.random.randn(10, 3)
        v2 = rf_world.transform_points(v1, rf_child1)

        t, r = best_fit_transform(v1, v2)
        t_exp, r_exp, _ = rf_world.get_transformation(rf_child1)
        npt.assert_allclose(t, t_exp, rtol=1.0, atol=1e-10)
        npt.assert_allclose(np.abs(r), np.abs(r_exp), rtol=1.0, atol=1e-10)

    def test_best_fit_transform_xr(self, get_rf_tree, mock_quaternion):
        """"""
        xr = pytest.importorskip("xarray")

        rf_world, rf_child1, _ = get_rf_tree(
            (1.0, 0.0, 0.0), mock_quaternion(np.pi / 2, 0.0, 0.0)
        )
        v1 = np.random.randn(10, 3)
        v2 = rf_world.transform_points(v1, rf_child1)

        v1_da = xr.DataArray(
            v1, {"cartesian_axis": ["x", "y", "z"]}, ("time", "cartesian_axis")
        )
        t, r = best_fit_transform(v1_da.T, v2.T, dim="cartesian_axis")

        assert t.dims == ("cartesian_axis",)
        assert r.dims == ("quaternion_axis",)

    def test_iterative_closest_point(self, icp_test_data):
        """"""
        v1 = icp_test_data["t265"]
        v2 = icp_test_data["vicon"]

        t, r = iterative_closest_point(v1, v2)
        npt.assert_allclose(t, [-1.89872037, 0.61755277, 0.95930489])
        npt.assert_allclose(
            r,
            [-6.87968663e-01, 4.73801246e-04, 9.12595868e-03, 7.25682859e-01],
        )

    def test_iterative_closest_point_xr(self, get_rf_tree, mock_quaternion):
        """"""
        xr = pytest.importorskip("xarray")

        rf_world, rf_child1, _ = get_rf_tree(
            (1.0, 0.0, 0.0), mock_quaternion(np.pi / 2, 0.0, 0.0)
        )
        v1 = np.random.randn(10, 3)
        v2 = rf_world.transform_points(v1, rf_child1)

        v1_da = xr.DataArray(
            v1, {"cartesian_axis": ["x", "y", "z"]}, ("time", "cartesian_axis")
        )
        t, r = iterative_closest_point(
            v1_da.T,
            v2.T,
            dim="cartesian_axis",
            init_transform=((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        )

        assert t.dims == ("cartesian_axis",)
        assert r.dims == ("quaternion_axis",)
