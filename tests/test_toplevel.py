import numpy as np
import pytest
from numpy import testing as npt

import rigid_body_motion as rbm


class TestTopLevel:
    """"""

    @pytest.fixture()
    def rf_tree(self, register_rf_tree, mock_quaternion):
        """"""
        register_rf_tree(
            tc1=(1.0, 0.0, 0.0),
            tc2=(-1.0, 0.0, 0.0),
            rc2=mock_quaternion(np.pi, 0.0, 0.0),
        )

    def test_example_data(self):
        """"""
        xr = pytest.importorskip("xarray")
        pytest.importorskip("netCDF4")
        pytest.importorskip("pooch")

        xr.load_dataset(rbm.example_data["head"])
        xr.load_dataset(rbm.example_data["left_eye"])
        xr.load_dataset(rbm.example_data["right_eye"])

        assert rbm.example_data["rosbag"].exists()

    def test_transform_points(self, rf_tree):
        """"""
        arr_child2 = (1.0, 1.0, 1.0)
        arr_exp = (-3.0, -1.0, 1.0)

        # tuple
        arr_child1 = rbm.transform_points(
            arr_child2, into="child1", outof="child2"
        )
        npt.assert_almost_equal(arr_child1, arr_exp)

    def test_transform_points_xr(self, rf_tree):
        """"""
        xr = pytest.importorskip("xarray")

        arr_child2 = (1.0, 1.0, 1.0)
        arr_exp = (-3.0, -1.0, 1.0)

        da_child2 = xr.DataArray(
            np.tile(arr_child2, (10, 1)),
            {"time": np.arange(10)},
            ("time", "cartesian_axis"),
        )
        da_child1 = rbm.transform_points(
            da_child2,
            into="child1",
            outof="child2",
            dim="cartesian_axis",
            timestamps="time",
        )
        assert da_child1.shape == (10, 3)
        npt.assert_almost_equal(da_child1[0], arr_exp)

        # multi-dimensional vectors
        da_child2 = xr.DataArray(
            np.tile(arr_child2, (5, 10, 1)),
            {"time": np.arange(10)},
            ("extra_dim", "time", "cartesian_axis"),
            attrs={"reference_frame": "child2"},
        )
        da_child1 = rbm.transform_points(
            da_child2, into="child1", dim="cartesian_axis", timestamps="time",
        )
        assert da_child1.shape == (5, 10, 3)
        assert da_child1.dims == ("extra_dim", "time", "cartesian_axis")
        assert da_child1.attrs["reference_frame"] == "child1"
        assert da_child1.attrs["representation_frame"] == "child1"
        npt.assert_almost_equal(da_child1[0, 0], arr_exp)

    def test_transform_quaternions(self, rf_tree, mock_quaternion):
        """"""
        arr_child2 = (1.0, 0.0, 0.0, 0.0)
        arr_exp = mock_quaternion(np.pi, 0.0, 0.0)

        # tuple
        arr_child1 = rbm.transform_quaternions(
            arr_child2, into="child1", outof="child2"
        )
        npt.assert_almost_equal(arr_child1, arr_exp)

    def test_transform_quaternions_xr(self, rf_tree, mock_quaternion):
        """"""
        xr = pytest.importorskip("xarray")

        arr_child2 = (1.0, 0.0, 0.0, 0.0)
        arr_exp = mock_quaternion(np.pi, 0.0, 0.0)

        da_child2 = xr.DataArray(
            np.tile(arr_child2, (10, 1)),
            {"time": np.arange(10)},
            ("time", "quaternion_axis"),
            attrs={"reference_frame": "child2"},
        )
        da_child1 = rbm.transform_quaternions(
            da_child2, into="child1", dim="quaternion_axis", timestamps="time",
        )
        assert da_child1.shape == (10, 4)
        assert da_child1.attrs["reference_frame"] == "child1"
        assert da_child1.attrs["representation_frame"] == "child1"
        npt.assert_almost_equal(da_child1[0], arr_exp)

        # multi-dimensional vectors
        da_child2 = xr.DataArray(
            np.tile(arr_child2, (5, 10, 1)),
            {"time": np.arange(10)},
            ("extra_dim", "time", "quaternion_axis"),
        )
        da_child1 = rbm.transform_quaternions(
            da_child2,
            into="child1",
            outof="child2",
            dim="quaternion_axis",
            timestamps="time",
        )
        assert da_child1.shape == (5, 10, 4)
        assert da_child1.dims == ("extra_dim", "time", "quaternion_axis")
        npt.assert_almost_equal(da_child1[0, 0], arr_exp)

    def test_scalar_dataarray(self, head_rf_tree):
        """"""
        xr = pytest.importorskip("xarray")
        g_world = xr.DataArray(
            [0.0, 1.0, 0.0],
            {"cartesian_axis": ["x", "y", "z"]},
            "cartesian_axis",
        )
        q_world = xr.DataArray(
            [1.0, 0.0, 0.0, 0.0],
            {"quaternion_axis": ["w", "x", "y", "z"]},
            "quaternion_axis",
        )
        rbm.transform_vectors(g_world, outof="world", into="head")
        rbm.transform_points(g_world, outof="world", into="head")
        rbm.transform_quaternions(q_world, outof="world", into="head")

    def test_transform_vectors(self, rf_tree):
        """"""
        arr_child2 = (1.0, 1.0, 1.0)
        arr_exp = (-1.0, -1.0, 1.0)

        # tuple
        arr_child1 = rbm.transform_vectors(
            arr_child2, into="child1", outof="child2"
        )
        npt.assert_almost_equal(arr_child1, arr_exp)

    def test_transform_vectors_xr(self, rf_tree):
        """"""
        xr = pytest.importorskip("xarray")

        arr_child2 = (1.0, 1.0, 1.0)
        arr_exp = (-1.0, -1.0, 1.0)

        da_child2 = xr.DataArray(
            np.tile(arr_child2, (10, 1)),
            {"time": np.arange(10)},
            ("time", "cartesian_axis"),
            attrs={"representation_frame": "child2"},
        )
        da_child1 = rbm.transform_vectors(
            da_child2, into="child1", dim="cartesian_axis", timestamps="time",
        )
        assert da_child1.shape == (10, 3)
        assert da_child1.attrs["representation_frame"] == "child1"
        npt.assert_almost_equal(da_child1[0], arr_exp)

        # multi-dimensional vectors
        da_child2 = xr.DataArray(
            np.tile(arr_child2, (5, 10, 1)),
            {"time": np.arange(10)},
            ("extra_dim", "time", "cartesian_axis"),
        )
        da_child1 = rbm.transform_vectors(
            da_child2,
            into="child1",
            outof="child2",
            dim="cartesian_axis",
            timestamps="time",
        )
        assert da_child1.shape == (5, 10, 3)
        assert da_child1.dims == ("extra_dim", "time", "cartesian_axis")
        npt.assert_almost_equal(da_child1[0, 0], arr_exp)

    def test_transform_angular_velocity_xr(self, compensated_tree):
        """"""
        pytest.importorskip("xarray")

        head_twist = rbm.lookup_twist(
            "head", represent_in="head", as_dataset=True
        )
        gaze_twist = rbm.lookup_twist("eyes", as_dataset=True)

        eye_angular_rf = rbm.transform_angular_velocity(
            gaze_twist.angular_velocity,
            outof="head",
            into="world",
            timestamps="time",
        )

        assert (eye_angular_rf < 1e-10).all()

        eye_angular_mf = rbm.transform_angular_velocity(
            head_twist.angular_velocity,
            outof="head",
            into="eyes",
            what="moving_frame",
            timestamps="time",
        )

        assert (eye_angular_mf < 1e-10).all()

    def test_transform_linear_velocity_xr(self, compensated_tree):
        """"""
        pytest.importorskip("xarray")

        head_twist = rbm.lookup_twist(
            "head", represent_in="head", as_dataset=True
        )
        gaze_twist = rbm.lookup_twist("eyes", as_dataset=True)

        eye_linear_rf = rbm.transform_linear_velocity(
            gaze_twist.linear_velocity[1:-1],
            outof="head",
            into="world",
            moving_frame="eyes",
            timestamps="time",
        )

        assert (eye_linear_rf < 0.06).all()

        eye_linear_mf = rbm.transform_linear_velocity(
            head_twist.linear_velocity[1:-1],
            outof="head",
            into="eyes",
            what="moving_frame",
            reference_frame="world",
            timestamps="time",
        )

        assert (eye_linear_mf < 0.06).all()

    def test_transform_coordinates(self):
        """"""
        # ndarray
        arr = np.ones((10, 2))
        expected = np.tile((np.sqrt(2), np.pi / 4), (10, 1))
        actual = rbm.transform_coordinates(
            arr, into="polar", outof="cartesian", axis=1
        )
        npt.assert_equal(actual, expected)

        with pytest.raises(ValueError):
            rbm.transform_coordinates(
                np.ones((10, 3)), into="polar", outof="cartesian"
            )
        with pytest.raises(ValueError):
            rbm.transform_coordinates(
                np.ones((10, 2)), into="polar", outof="unsupported"
            )

    def test_transform_coordinates_xr(self):
        """"""
        xr = pytest.importorskip("xarray")

        # DataArray
        da = xr.DataArray(
            np.ones((10, 3)),
            {"time": np.arange(10), "cartesian_axis": ["x", "y", "z"]},
            ("time", "cartesian_axis"),
        )
        expected = xr.DataArray(
            np.tile((1.732051, 0.955317, 0.785398), (10, 1)),
            {"time": np.arange(10), "spherical_axis": ["r", "theta", "phi"]},
            ("time", "spherical_axis"),
        )
        actual = rbm.transform_coordinates(
            da, into="spherical", outof="cartesian"
        )
        xr.testing.assert_allclose(actual, expected)

        # source coordinate system in attrs:
        da.attrs = {"coordinate_system": "cartesian"}
        actual = rbm.transform_coordinates(da, into="spherical")
        xr.testing.assert_allclose(actual, expected)
        assert actual.attrs == {"coordinate_system": "spherical"}

        # regression test for time axis being erroneously inferred
        actual = rbm.transform_coordinates(
            da.T, into="spherical", outof="cartesian", dim="cartesian_axis"
        )
        xr.testing.assert_allclose(actual, expected.T)

        da.attrs = {}
        with pytest.raises(ValueError):
            rbm.transform_coordinates(da, into="spherical")

    def test_lookup_transform_xr(self, rf_tree, head_dataset):
        """"""
        xr = pytest.importorskip("xarray")

        # static frame
        transform = rbm.lookup_transform("world", "child2", as_dataset=True)
        assert set(transform.dims) == {"cartesian_axis", "quaternion_axis"}
        np.testing.assert_almost_equal(
            transform.translation.values, (-1.0, 0.0, 0.0)
        )
        np.testing.assert_almost_equal(
            transform.rotation.values, (0.0, 0.0, 0.0, -1.0)
        )

        # moving frame
        rbm.register_frame("world", update=True)
        rbm.ReferenceFrame.from_dataset(
            head_dataset, "position", "orientation", "time", "world", "head",
        ).register(update=True)

        transform = rbm.lookup_transform("head", "world", as_dataset=True)
        xr.testing.assert_allclose(
            transform.translation, head_dataset.position
        )
        assert transform.translation.attrs["reference_frame"] == "head"
        assert transform.translation.attrs["representation_frame"] == "head"
        xr.testing.assert_allclose(
            transform.rotation, head_dataset.orientation
        )
        assert transform.rotation.attrs["reference_frame"] == "head"
        assert transform.rotation.attrs["representation_frame"] == "head"

    def test_lookup_pose_xr(self, rf_tree, head_dataset):
        """"""
        xr = pytest.importorskip("xarray")

        # static frame
        pose = rbm.lookup_pose("world", "child2", as_dataset=True)
        assert set(pose.dims) == {"cartesian_axis", "quaternion_axis"}
        np.testing.assert_almost_equal(pose.position.values, (-1.0, 0.0, 0.0))
        np.testing.assert_almost_equal(
            pose.orientation.values, (0.0, 0.0, 0.0, -1.0)
        )

        # moving frame
        rbm.register_frame("world", update=True)
        rbm.ReferenceFrame.from_dataset(
            head_dataset, "position", "orientation", "time", "world", "head",
        ).register(update=True)

        pose = rbm.lookup_pose("head", "world", as_dataset=True)
        xr.testing.assert_allclose(pose.position, head_dataset.position)
        assert pose.position.attrs["reference_frame"] == "world"
        assert pose.position.attrs["representation_frame"] == "world"
        xr.testing.assert_allclose(pose.orientation, head_dataset.orientation)
        assert pose.orientation.attrs["reference_frame"] == "world"
        assert pose.orientation.attrs["representation_frame"] == "world"

    def test_lookup_twist_xr(self, head_dataset):
        """"""
        rbm.register_frame("world")
        rbm.ReferenceFrame.from_dataset(
            head_dataset, "position", "orientation", "time", "world", "head",
        ).register(update=True)

        twist = rbm.lookup_twist(
            "head",
            "world",
            "world",
            outlier_thresh=1e-3,
            cutoff=0.25,
            as_dataset=True,
        )

        err_v = (
            twist.linear_velocity
            - head_dataset.interp(time=twist.time).linear_velocity
        ) ** 2
        assert err_v.mean() < 1e-4

        err_w = (
            twist.angular_velocity
            - head_dataset.interp(time=twist.time).angular_velocity
        ) ** 2
        assert err_w.mean() < 1e-3

    def test_lookup_linear_velocity_xr(self, head_dataset):
        """"""
        rbm.register_frame("world")
        rbm.ReferenceFrame.from_dataset(
            head_dataset, "position", "orientation", "time", "world", "head",
        ).register(update=True)

        da = rbm.lookup_linear_velocity(
            "head",
            "world",
            "world",
            outlier_thresh=1e-3,
            cutoff=0.25,
            as_dataarray=True,
        )

        err_v = (da - head_dataset.linear_velocity.interp(time=da.time)) ** 2
        assert err_v.mean() < 1e-4

    def test_lookup_angular_velocity_xr(self, head_dataset):
        """"""
        rbm.register_frame("world")
        rbm.ReferenceFrame.from_dataset(
            head_dataset, "position", "orientation", "time", "world", "head",
        ).register(update=True)

        da = rbm.lookup_angular_velocity(
            "head", "world", "world", cutoff=0.25, as_dataarray=True,
        )

        err_v = (da - head_dataset.angular_velocity.interp(time=da.time)) ** 2
        assert err_v.mean() < 1e-3
