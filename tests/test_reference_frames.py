import numpy as np
import pandas as pd
import pytest
from numpy import testing as npt

import rigid_body_motion as rbm
from rigid_body_motion.reference_frames import _deregister, _register


class TestReferenceFrameRegistry:
    def test_register(self):
        """"""
        rf_world = rbm.ReferenceFrame("world")
        _register(rf_world)
        assert rbm.registry["world"] is rf_world

        # name is None
        with pytest.raises(ValueError):
            _register(rbm.ReferenceFrame())

        # already registered
        with pytest.raises(ValueError):
            _register(rf_world)

        # update=True
        rf_child = rbm.ReferenceFrame("child", parent=rf_world)
        _register(rf_child)
        rf_child_new = rbm.ReferenceFrame("child", parent=rf_world)
        _register(rf_child_new, update=True)
        assert rbm.registry["child"] is rf_child_new
        assert rbm.registry["child"].parent is rf_world
        assert rbm.registry["world"].children == (rf_child_new,)

    def test_deregister(self):
        """"""
        rbm.register_frame("world")
        _deregister("world")
        assert "world" not in rbm.registry

        with pytest.raises(ValueError):
            _deregister("not_an_rf")

    def test_register_frame(self):
        """"""
        rbm.register_frame("world")
        assert isinstance(rbm.registry["world"], rbm.ReferenceFrame)

    def test_deregister_frame(self):
        """"""
        rbm.register_frame("world")
        rbm.deregister_frame("world")
        assert "world" not in rbm.registry

    def test_clear_registry(self):
        """"""
        rbm.register_frame("world")
        rbm.clear_registry()
        assert len(rbm.registry) == 0

    def test_render_tree(self, register_rf_tree, capfd):
        """"""
        register_rf_tree()
        rbm.render_tree("world")
        out, err = capfd.readouterr()
        assert out == "world\n├── child1\n└── child2\n"


class TestReferenceFrame:
    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """"""
        rbm.clear_registry()
        yield

    def test_constructor(self):
        """"""
        rf_world = rbm.ReferenceFrame("world")
        rf_child = rbm.ReferenceFrame("child", parent=rf_world)
        assert rf_child.parent is rf_world

        _register(rf_world)
        rf_child2 = rbm.ReferenceFrame("child2", parent="world")
        assert rf_child2.parent is rf_world

        # invalid parent
        with pytest.raises(ValueError):
            rbm.ReferenceFrame("child3", parent="not_an_rf")

    def test_destructor(self):
        """"""
        rf_world = rbm.ReferenceFrame("world")
        _register(rf_world)
        del rf_world
        assert "world" in rbm.registry
        del rbm.registry["world"]
        assert "world" not in rbm.registry

    def test_str(self):
        """"""
        rf_world = rbm.ReferenceFrame("world")
        assert str(rf_world) == "<ReferenceFrame 'world'>"

    def test_init_arrays(self):
        """"""
        # tuples
        t, r, ts = rbm.ReferenceFrame._init_arrays(
            (1.0, 1.0, 1.0), (1.0, 0.0, 0.0, 0.0), None, False
        )
        assert isinstance(t, np.ndarray)
        assert isinstance(r, np.ndarray)
        assert ts is None

        # nothing
        t, r, ts = rbm.ReferenceFrame._init_arrays(None, None, None, False)
        npt.assert_equal(t, (0.0, 0.0, 0.0))
        npt.assert_equal(r, (1.0, 0.0, 0.0, 0.0))
        assert ts is None

        # timestamps
        t, r, ts = rbm.ReferenceFrame._init_arrays(
            np.ones((10, 3)), np.ones((10, 4)), np.arange(10), False
        )
        npt.assert_equal(t, np.ones((10, 3)))
        npt.assert_equal(r, np.ones((10, 4)))
        npt.assert_equal(ts, np.arange(10))

        # timestamps not 1d
        with pytest.raises(ValueError):
            rbm.ReferenceFrame._init_arrays(None, None, np.ones((5, 5)), False)

        # wrong r/t shape
        with pytest.raises(ValueError):
            rbm.ReferenceFrame._init_arrays(np.ones((5, 3)), None, None, False)
        with pytest.raises(ValueError):
            rbm.ReferenceFrame._init_arrays(None, np.ones((5, 3)), None, False)

        # inverse
        t, r, ts = rbm.ReferenceFrame._init_arrays(
            (1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0), None, True
        )
        npt.assert_allclose(t, (1.0, 1.0, -1.0))
        npt.assert_allclose(r, (0.0, 0.0, 0.0, -1.0))
        assert ts is None

    def test_register(self):
        """"""
        rf_world = rbm.ReferenceFrame("world")
        rf_world.register()
        assert "world" in rbm.registry
        assert rbm.registry["world"] is rf_world

        # update=True
        rf_world2 = rbm.ReferenceFrame("world")
        rf_world2.register(update=True)
        assert rbm.registry["world"] is rf_world2

    def test_deregister(self):
        """"""
        rf_world = rbm.ReferenceFrame("world")
        rf_world.register()
        rf_world.deregister()
        assert "world" not in rbm.registry

    def test_walk(self):
        """"""
        rf_world = rbm.ReferenceFrame("world")
        rf_child = rbm.ReferenceFrame("child", parent=rf_world)
        rf_child2 = rbm.ReferenceFrame("child2", parent=rf_world)

        up, down = rf_child._walk(rf_child2)
        assert up == (rf_child,)
        assert down == (rf_child2,)

    def test_validate_input(self):
        """"""
        # scalar input
        arr_scalar = (0.0, 0.0, 0.0)
        arr_val, _ = rbm.ReferenceFrame._validate_input(
            arr_scalar, -1, 3, None, 0
        )
        assert isinstance(arr_val, np.ndarray)
        npt.assert_equal(arr_val, np.zeros(3))

        # wrong axis length
        with pytest.raises(ValueError):
            rbm.ReferenceFrame._validate_input(arr_scalar, -1, 4, None, 0)

        # array with timestamps
        arr = np.ones((10, 3))
        timestamps = range(10)
        arr_val, ts_val = rbm.ReferenceFrame._validate_input(
            arr, -1, 3, timestamps, 0
        )
        assert isinstance(ts_val, np.ndarray)

        # time axis not first axis
        arr = np.ones((5, 10, 3))
        timestamps = range(10)
        arr_val, ts_val = rbm.ReferenceFrame._validate_input(
            arr, -1, 3, timestamps, 1
        )
        assert arr_val.shape == (10, 5, 3)

        # timestamps not 1D
        with pytest.raises(ValueError):
            rbm.ReferenceFrame._validate_input(
                arr, -1, 3, np.ones((10, 10)), 0
            )

        # first axis doesn't match timestamps
        with pytest.raises(ValueError):
            rbm.ReferenceFrame._validate_input(arr[:-1], -1, 3, timestamps, 0)

    def test_expand_singleton_axes(self):
        """"""
        # single translation
        t = np.zeros(3)
        assert rbm.ReferenceFrame._expand_singleton_axes(t, 2).shape == (3,)

        # multiple translations
        t = np.zeros((10, 3))
        assert rbm.ReferenceFrame._expand_singleton_axes(t, 2).shape == (10, 3)
        assert rbm.ReferenceFrame._expand_singleton_axes(t, 3).shape == (
            10,
            1,
            3,
        )

    def test_from_dataset(self):
        """"""
        xr = pytest.importorskip("xarray")
        ds = xr.Dataset(
            {
                "t": (["time", "cartesian_axis"], np.ones((10, 3))),
                "r": (["time", "quaternion_axis"], np.ones((10, 4))),
            },
            {"time": np.arange(10)},
        )

        rf_world = rbm.ReferenceFrame("world")
        rf_child = rbm.ReferenceFrame.from_dataset(
            ds, "t", "r", "time", rf_world
        )

        npt.assert_equal(rf_child.translation, np.ones((10, 3)))
        npt.assert_equal(rf_child.rotation, np.ones((10, 4)))
        npt.assert_equal(rf_child.timestamps, np.arange(10))

    def test_from_translation_datarray(self):
        """"""
        xr = pytest.importorskip("xarray")
        da = xr.DataArray(
            np.ones((10, 3)), {"time": np.arange(10)}, ("time", "axis")
        )

        rf_world = rbm.ReferenceFrame("world")
        rf_child = rbm.ReferenceFrame.from_translation_dataarray(
            da, "time", rf_world
        )

        npt.assert_equal(rf_child.translation, np.ones((10, 3)))
        npt.assert_equal(rf_child.timestamps, np.arange(10))

    def test_from_rotation_datarray(self):
        """"""
        xr = pytest.importorskip("xarray")
        da = xr.DataArray(
            np.ones((10, 4)), {"time": np.arange(10)}, ("time", "axis")
        )

        rf_world = rbm.ReferenceFrame("world")
        rf_child = rbm.ReferenceFrame.from_rotation_dataarray(
            da, "time", rf_world
        )

        npt.assert_equal(rf_child.rotation, np.ones((10, 4)))
        npt.assert_equal(rf_child.timestamps, np.arange(10))

    def test_from_rotation_matrix(self):
        """"""
        mat = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        rf_world = rbm.ReferenceFrame("world")
        rf_child = rbm.ReferenceFrame.from_rotation_matrix(mat, rf_world)

        npt.assert_equal(rf_child.translation, (0.0, 0.0, 0.0))
        npt.assert_allclose(rf_child.rotation, (-0.5, -0.5, 0.5, 0.5))
        assert rf_child.timestamps is None

        with pytest.raises(ValueError):
            rbm.ReferenceFrame.from_rotation_matrix(np.zeros((4, 4)), rf_world)

    def test_match_arrays(self):
        """"""
        arr1 = np.ones((10, 3))
        ts1 = np.arange(10)
        arr2 = np.ones((5, 3))
        ts2 = np.arange(5) + 2.5

        # float timestamps
        arr1_out, arr2_out, ts_out = rbm.ReferenceFrame._match_arrays(
            [(arr1, ts1), (arr2, ts2)],
        )
        npt.assert_allclose(arr1_out, arr1[:4])
        npt.assert_allclose(arr2_out, arr2[:4])
        npt.assert_equal(ts_out, ts1[3:7])

        # datetime timestamps
        ts1 = pd.date_range(start=0, freq="1s", periods=10).values
        ts2 = pd.date_range(start=0, freq="1s", periods=5).values
        arr1_out, arr2_out, ts_out = rbm.ReferenceFrame._match_arrays(
            [(arr1, ts1), (arr2, ts2)],
        )
        npt.assert_allclose(arr1_out, arr1[:5])
        npt.assert_allclose(arr2_out, arr2)
        npt.assert_allclose(ts_out.astype(float), ts1[:5].astype(float))
        assert ts_out.dtype == ts1.dtype

    def test_lookup_transform(self, rf_grid, get_rf_tree):
        """"""
        r, rc1, rc2, t, tc1, tc2 = rf_grid
        rf_world, rf_child1, rf_child2 = get_rf_tree(tc1, rc1, tc2, rc2)

        # child1 to world
        t_act, r_act, ts = rf_child1.lookup_transform(rf_world)
        npt.assert_almost_equal(t_act, tc1)
        npt.assert_almost_equal(r_act, rc1)
        assert ts is None

        # child1 to child2
        t_act, r_act, ts = rf_child1.lookup_transform(rf_child2)
        npt.assert_almost_equal(t_act, t)
        npt.assert_almost_equal(r_act, r)
        assert ts is None

        # inverse
        rf_child1_inv = rbm.ReferenceFrame(
            parent=rf_world, translation=tc1, rotation=rc1, inverse=True
        )
        t_act, r_act, ts = rf_world.lookup_transform(rf_child1_inv)
        npt.assert_almost_equal(t_act, tc1)
        npt.assert_almost_equal(r_act, rc1)
        assert ts is None

        # with timestamps
        rf_child3 = rbm.ReferenceFrame(
            "child3", rf_child1, timestamps=np.arange(5) + 2.5
        )
        rf_child4 = rbm.ReferenceFrame(
            "child4", rf_child2, timestamps=np.arange(10)
        )

        t_act, r_act, ts = rf_child3.lookup_transform(rf_child4)
        npt.assert_almost_equal(t_act, np.tile(t, (5, 1)))
        npt.assert_almost_equal(r_act, np.tile(r, (5, 1)))
        npt.assert_equal(ts, np.arange(5) + 2.5)

    def test_lookup_transform_discrete(self):
        """"""
        t1 = np.ones((10, 3))
        t2 = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

        rf_world = rbm.ReferenceFrame("world")
        rf_child1 = rbm.ReferenceFrame(
            "child1", rf_world, translation=t1, timestamps=np.arange(10),
        )
        rf_child2 = rbm.ReferenceFrame(
            "child2",
            rf_world,
            translation=t2,
            timestamps=[2, 5],
            discrete=True,
        )

        # interpolated first
        t_act, r_act, ts = rf_child1.lookup_transform(rf_child2)
        npt.assert_equal(t_act, [[0.0, 1.0, 1.0]] * 5 + [[-1.0, 1.0, 1.0]] * 5)
        npt.assert_equal(r_act, np.tile([[1.0, 0.0, 0.0, 0.0]], (10, 1)))
        npt.assert_allclose(ts, np.arange(10))

        # event-based first
        t_act, r_act, ts = rf_child2.lookup_transform(rf_child1)
        npt.assert_equal(
            t_act, [[0.0, -1.0, -1.0]] * 5 + [[1.0, -1.0, -1.0]] * 5
        )
        npt.assert_equal(r_act, np.tile([[1.0, 0.0, 0.0, 0.0]], (10, 1)))
        npt.assert_allclose(ts, np.arange(10))

    def test_transform_vectors(self, transform_grid, get_rf_tree):
        """"""
        o, ot, p, pt, rc1, rc2, tc1, tc2 = transform_grid
        _, rf_child1, rf_child2 = get_rf_tree(tc1, rc1, tc2, rc2)
        rf_child3 = rbm.ReferenceFrame(
            "child3", rf_child1, timestamps=np.arange(5) + 2.5
        )

        # static reference frame + single vector
        vt_act = rf_child1.transform_vectors(p, rf_child2)
        v0t = rf_child1.transform_points((0.0, 0.0, 0.0), rf_child2)
        vt = np.array(pt) - np.array(v0t)
        np.testing.assert_allclose(vt_act, vt, rtol=1.0, atol=1e-15)

        # moving reference frame + single vector
        vt_act = rf_child3.transform_vectors(p, rf_child2)
        v0t = rf_child3.transform_points((0.0, 0.0, 0.0), rf_child2)
        vt = np.tile(pt, (5, 1)) - np.array(v0t)
        np.testing.assert_allclose(vt_act, vt, rtol=1.0, atol=1e-15)

        # moving reference frame + multiple vectors
        vt_act = rf_child3.transform_vectors(
            np.tile(p, (10, 1)), rf_child2, timestamps=np.arange(10)
        )
        v0t = rf_child3.transform_points(
            np.tile((0.0, 0.0, 0.0), (10, 1)),
            rf_child2,
            timestamps=np.arange(10),
        )
        vt = np.tile(pt, (4, 1)) - np.array(v0t)
        np.testing.assert_allclose(vt_act, vt, rtol=1.0, atol=1e-15)

        # moving reference frame + multiple n-dimensional vectors
        vt_act = rf_child3.transform_vectors(
            np.tile(p, (10, 10, 1)),
            rf_child2,
            timestamps=np.arange(10),
            time_axis=1,
        )
        v0t = rf_child3.transform_points(
            np.tile((0.0, 0.0, 0.0), (10, 1)),
            rf_child2,
            timestamps=np.arange(10),
        )
        vt = np.tile(pt, (10, 4, 1)) - np.array(v0t[np.newaxis, :, :])
        np.testing.assert_allclose(vt_act, vt, rtol=1.0, atol=1e-15)

    def test_transform_points(self, transform_grid, get_rf_tree):
        """"""
        o, ot, p, pt, rc1, rc2, tc1, tc2 = transform_grid
        rf_world, rf_child1, rf_child2 = get_rf_tree(tc1, rc1, tc2, rc2)
        rf_child3 = rbm.ReferenceFrame(
            "child3", rf_child1, timestamps=np.arange(5) + 2.5
        )

        # static reference frame + single point
        pt_act = rf_child1.transform_points(p, rf_child2)
        np.testing.assert_allclose(pt_act, pt)

        # moving reference frame + single point
        pt_act = rf_child3.transform_points(p, rf_child2)
        np.testing.assert_allclose(pt_act, np.tile(pt, (5, 1)), rtol=1.0)

        # moving reference frame + multiple points
        pt_act = rf_child3.transform_points(
            np.tile(p, (10, 1)), rf_child2, timestamps=np.arange(10)
        )
        np.testing.assert_allclose(pt_act, np.tile(pt, (4, 1)), rtol=1.0)

        # moving reference frame + multiple n-dimensional points
        pt_act = rf_child3.transform_points(
            np.tile(p, (10, 10, 1)),
            rf_child2,
            timestamps=np.arange(10),
            time_axis=1,
        )
        np.testing.assert_allclose(pt_act, np.tile(pt, (10, 4, 1)), rtol=1.0)

    def test_transform_quaternions(self, transform_grid, get_rf_tree):
        """"""
        o, ot, p, pt, rc1, rc2, tc1, tc2 = transform_grid
        _, rf_child1, rf_child2 = get_rf_tree(tc1, rc1, tc2, rc2)
        rf_child3 = rbm.ReferenceFrame(
            "child3", rf_child1, timestamps=np.arange(5) + 2.5
        )

        # static reference frame + single quaternion
        ot_act = rf_child1.transform_quaternions(o, rf_child2)
        npt.assert_allclose(np.abs(ot_act), np.abs(ot), rtol=1.0)

        # moving reference frame + single quaternion
        ot_act = rf_child3.transform_quaternions(o, rf_child2)
        np.testing.assert_allclose(
            np.abs(ot_act), np.tile(np.abs(ot), (5, 1)), rtol=1.0
        )

        # moving reference frame + multiple vectors
        ot_act = rf_child3.transform_quaternions(
            np.tile(o, (10, 1)), rf_child2, timestamps=np.arange(10)
        )
        np.testing.assert_allclose(
            np.abs(ot_act), np.tile(np.abs(ot), (4, 1)), rtol=1.0
        )

        # moving reference frame + multiple n-dimensional points
        ot_act = rf_child3.transform_quaternions(
            np.tile(o, (10, 10, 1)),
            rf_child2,
            timestamps=np.arange(10),
            time_axis=1,
        )
        np.testing.assert_allclose(
            np.abs(ot_act), np.tile(np.abs(ot), (10, 4, 1)), rtol=1.0
        )

    def test_lookup_twist(self, compensated_tree):
        """"""
        _, w_head_world = rbm.registry["head"].lookup_twist()
        _, w_eyes_head = rbm.registry["eyes"].lookup_twist()
        v_eyes_world, w_eyes_world = rbm.registry["eyes"].lookup_twist("world")

        npt.assert_allclose(w_head_world, -w_eyes_head, rtol=0.1, atol=1e-10)
        assert (v_eyes_world < 1e-10).all()
        assert (w_eyes_world < 1e-10).all()

    def test_transform_angular_velocity(self, compensated_tree):
        """"""
        _, w_head_world, ts = rbm.registry["head"].lookup_twist(
            return_timestamps=True
        )
        _, w_eyes_head, ts = rbm.registry["eyes"].lookup_twist(
            return_timestamps=True, represent_in="eyes"
        )

        # transform reference frame
        w_eyes_world_rf = rbm.registry["head"].transform_angular_velocity(
            w_eyes_head, "world", timestamps=ts,
        )
        assert (w_eyes_world_rf < 1e-10).all()

        # transform moving frame
        w_eyes_world_mf = rbm.registry["head"].transform_angular_velocity(
            w_head_world, "eyes", what="moving_frame", timestamps=ts,
        )
        assert (w_eyes_world_mf < 1e-10).all()

    def test_transform_linear_velocity(self, compensated_tree):
        """"""
        v_head_world, _, ts = rbm.registry["head"].lookup_twist(
            return_timestamps=True, represent_in="head"
        )
        v_eyes_head, _, ts = rbm.registry["eyes"].lookup_twist(
            return_timestamps=True
        )

        # transform reference frame
        v_eyes_world_rf = rbm.registry["head"].transform_linear_velocity(
            v_eyes_head[1:-1],
            "world",
            moving_frame="eyes",
            timestamps=ts[1:-1],
        )
        assert (np.abs(v_eyes_world_rf) < 1e-3).all()

        # transform moving frame
        v_eyes_world_mf = rbm.registry["head"].transform_linear_velocity(
            v_head_world[1:-1],
            "eyes",
            what="moving_frame",
            reference_frame="world",
            timestamps=ts[1:-1],
        )
        assert (np.abs(v_eyes_world_mf) < 1e-3).all()
