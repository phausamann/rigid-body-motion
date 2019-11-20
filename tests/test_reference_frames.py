import pytest
from numpy import testing as npt
from .helpers import rf_test_grid, transform_test_grid, get_rf_tree

import numpy as np
import pandas as pd
import xarray as xr

import rigid_body_motion as rbm
from rigid_body_motion.reference_frames import _register, _deregister


class TestReferenceFrameRegistry(object):

    def test_register(self):
        """"""
        rf_world = rbm.ReferenceFrame('world')
        _register(rf_world)
        assert rbm._rf_registry['world'] is rf_world

        # name is None
        with pytest.raises(ValueError):
            _register(rbm.ReferenceFrame())

        # already registered
        with pytest.raises(ValueError):
            _register(rf_world)

        # update=True
        rf_world2 = rbm.ReferenceFrame('world')
        _register(rf_world2, update=True)
        assert rbm._rf_registry['world'] is rf_world2

    def test_deregister(self):
        """"""
        _deregister('world')
        assert 'world' not in rbm._rf_registry

        with pytest.raises(ValueError):
            _deregister('not_an_rf')

    def test_register_frame(self):
        """"""
        rbm.register_frame('world')
        assert isinstance(rbm._rf_registry['world'], rbm.ReferenceFrame)

    def test_deregister_frame(self):
        """"""
        rbm.deregister_frame('world')
        assert 'world' not in rbm._rf_registry

    def test_clear_registry(self):
        """"""
        rbm.register_frame('world')
        rbm.clear_registry()
        assert len(rbm._rf_registry) == 0


class TestReferenceFrame(object):

    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """"""
        rbm.clear_registry()
        yield

    def test_constructor(self):
        """"""
        rf_world = rbm.ReferenceFrame('world')
        rf_child = rbm.ReferenceFrame('child', parent=rf_world)
        assert rf_child.parent is rf_world

        _register(rf_world)
        rf_child2 = rbm.ReferenceFrame('child2', parent='world')
        assert rf_child2.parent is rf_world

        # invalid parent
        with pytest.raises(ValueError):
            rbm.ReferenceFrame('child3', parent='not_an_rf')

    def test_destructor(self):
        """"""
        rf_world = rbm.ReferenceFrame('world')
        _register(rf_world)
        del rf_world
        assert 'world' in rbm._rf_registry
        del rbm._rf_registry['world']
        assert 'world' not in rbm._rf_registry

    def test_register(self):
        """"""
        rf_world = rbm.ReferenceFrame('world')
        rf_world.register()
        assert 'world' in rbm._rf_registry
        assert rbm._rf_registry['world'] is rf_world

        # update=True
        rf_world2 = rbm.ReferenceFrame('world')
        rf_world2.register(update=True)
        assert rbm._rf_registry['world'] is rf_world2

    def test_deregister(self):
        """"""
        rf_world = rbm.ReferenceFrame('world')
        rf_world.register()
        rf_world.deregister()
        assert 'world' not in rbm._rf_registry

    def test_walk(self):
        """"""
        rf_world = rbm.ReferenceFrame('world')
        rf_child = rbm.ReferenceFrame('child', parent=rf_world)
        rf_child2 = rbm.ReferenceFrame('child2', parent=rf_world)

        up, down = rf_child._walk(rf_child2)
        assert up == (rf_child,)
        assert down == (rf_child2,)

    def test_validate_input(self):
        """"""
        # scalar input
        arr_scalar = (0., 0., 0.)
        arr_val, _ = rbm.ReferenceFrame._validate_input(
            arr_scalar, -1, 3, None)
        assert isinstance(arr_val, np.ndarray)
        npt.assert_equal(arr_val, np.zeros(3))

        # wrong axis length
        with pytest.raises(ValueError):
            rbm.ReferenceFrame._validate_input(arr_scalar, -1, 4, None)

        # array with timestamps
        arr = np.ones((10, 3))
        timestamps = range(10)
        arr_val, ts_val = rbm.ReferenceFrame._validate_input(
            arr, -1, 3, timestamps)
        assert isinstance(ts_val, np.ndarray)

        # timestamps not 1D
        with pytest.raises(ValueError):
            rbm.ReferenceFrame._validate_input(arr, -1, 3, np.ones((10, 10)))

        # first axis doesn't match timestamps
        with pytest.raises(ValueError):
            rbm.ReferenceFrame._validate_input(arr[:-1], -1, 3, timestamps)

    def test_from_dataset(self):
        """"""
        ds = xr.Dataset(
            {'t': (['time', 'cartesian_axis'], np.ones((10, 3))),
             'r': (['time', 'quaternion_axis'], np.ones((10, 4)))},
            {'time': np.arange(10)})

        rf_world = rbm.ReferenceFrame('world')
        rf_child = rbm.ReferenceFrame.from_dataset(
            ds, 't', 'r', 'time', rf_world)

        npt.assert_equal(rf_child.translation, np.ones((10, 3)))
        npt.assert_equal(rf_child.rotation, np.ones((10, 4)))
        npt.assert_equal(rf_child.timestamps, np.arange(10))

    def test_from_rotation_matrix(self):
        """"""
        mat = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])

        rf_world = rbm.ReferenceFrame('world')
        rf_child = rbm.ReferenceFrame.from_rotation_matrix(mat, rf_world)

        npt.assert_equal(rf_child.translation, (0., 0., 0.))
        npt.assert_allclose(rf_child.rotation, (-0.5, -0.5,  0.5,  0.5))
        assert rf_child.timestamps is None

        with pytest.raises(ValueError):
            rbm.ReferenceFrame.from_rotation_matrix(np.zeros((4, 4)), rf_world)

    def test_interpolate(self):
        """"""
        arr = np.ones((10, 3))

        # float timestamps
        ts1 = np.arange(10)
        ts2 = np.arange(5) + 2.5
        arr_int, _ = rbm.ReferenceFrame._interpolate(arr, ts1, ts2)
        npt.assert_allclose(arr_int, arr[:5])

        # target range greater than source range
        arr_int, _ = rbm.ReferenceFrame._interpolate(arr[:5], ts2, ts1)
        npt.assert_allclose(arr_int, arr[:4])

        # datetime timestamps
        ts1 = pd.DatetimeIndex(start=0, freq='1s', periods=10).values
        ts2 = pd.DatetimeIndex(start=0, freq='1s', periods=5).values
        arr_int, ts_out = rbm.ReferenceFrame._interpolate(arr, ts1, ts2)
        npt.assert_allclose(arr_int, arr[:5])
        assert ts_out.dtype == ts1.dtype

        # not sorted
        with pytest.raises(ValueError):
            rbm.ReferenceFrame._interpolate(arr, ts1[::-1], ts2)
        with pytest.raises(ValueError):
            rbm.ReferenceFrame._interpolate(arr, ts1, ts2[::-1])

    @pytest.mark.parametrize('r, rc1, rc2, t, tc1, tc2', rf_test_grid())
    def test_get_transformation(self, r, rc1, rc2, t, tc1, tc2):
        """"""
        rf_world, rf_child1, rf_child2 = get_rf_tree(tc1, rc1, tc2, rc2)

        # child1 to world
        t_act, r_act, ts = rf_child1.get_transformation(rf_world)
        npt.assert_almost_equal(t_act, tc1)
        npt.assert_almost_equal(r_act, rc1)
        assert ts is None

        # child1 to child2
        t_act, r_act, ts = rf_child1.get_transformation(rf_child2)
        npt.assert_almost_equal(t_act, t)
        npt.assert_almost_equal(r_act, r)
        assert ts is None

        # with timestamps
        rf_child3 = rbm.ReferenceFrame(
            'child3', rf_child1, timestamps=np.arange(5) + 2.5)
        rf_child4 = rbm.ReferenceFrame(
            'child4', rf_child2, timestamps=np.arange(10))

        t_act, r_act, ts = rf_child3.get_transformation(rf_child4)
        npt.assert_almost_equal(t_act, np.tile(t, (5, 1)))
        npt.assert_almost_equal(r_act, np.tile(r, (5, 1)))
        npt.assert_equal(ts, np.arange(5) + 2.5)

    @pytest.mark.parametrize('o, ot, p, pt, rc1, rc2, tc1, tc2',
                             transform_test_grid())
    def test_get_transformation_func(
            self, o, ot, p, pt, rc1, rc2, tc1, tc2):
        """"""
        rf_world, rf_child1, rf_child2 = get_rf_tree(tc1, rc1, tc2, rc2)

        f = rf_child1.get_transformation_func(rf_child2)
        f_inv = rf_child2.get_transformation_func(rf_child1)

        # single point/orientation
        pt_act = f(np.array(p))
        npt.assert_almost_equal(pt_act, pt)
        ot_act = f(np.array(o))
        npt.assert_almost_equal(np.abs(ot_act), np.abs(ot))
        pt_act, ot_act = f((np.array(p), np.array(o)))
        npt.assert_almost_equal(pt_act, pt)
        npt.assert_almost_equal(np.abs(ot_act), np.abs(ot))

        # inverse transformation
        p_act = f_inv(np.array(pt))
        npt.assert_almost_equal(p_act, p)
        o_act = f_inv(np.array(ot))
        npt.assert_almost_equal(np.abs(o_act), np.abs(o))

        # array of points/orientations
        pt_act = f(np.tile(np.array(p)[None, :, None], (10, 1, 5)), axis=1)
        pt_exp = np.tile(np.array(pt)[None, :, None], (10, 1, 5))
        npt.assert_almost_equal(pt_act, pt_exp)
        ot_act = f(np.tile(np.array(o)[None, :, None], (10, 1, 5)), axis=1)
        ot_exp = np.tile(np.array(ot)[None, :, None], (10, 1, 5))
        npt.assert_almost_equal(np.abs(ot_act), np.abs(ot_exp))

    @pytest.mark.parametrize('o, ot, p, pt, rc1, rc2, tc1, tc2',
                             transform_test_grid())
    def test_transform_vectors(self, o, ot, p, pt, rc1, rc2, tc1, tc2):
        """"""
        _, rf_child1, rf_child2 = get_rf_tree(tc1, rc1, tc2, rc2)
        rf_child3 = rbm.ReferenceFrame(
            'child3', rf_child1, timestamps=np.arange(5) + 2.5)

        # static reference frame + single vector
        vt_act = rf_child1.transform_vectors(p, rf_child2)
        v0t = rf_child1.transform_points((0., 0., 0.), rf_child2)
        vt = np.array(pt) - np.array(v0t)
        np.testing.assert_allclose(vt_act, vt, rtol=1.)

        # moving reference frame + single vector
        vt_act = rf_child3.transform_vectors(p, rf_child2)
        v0t = rf_child3.transform_points((0., 0., 0.), rf_child2)
        vt = np.tile(pt, (5, 1)) - np.array(v0t)
        np.testing.assert_allclose(vt_act, vt, rtol=1.)

        # moving reference frame + multiple vectors
        vt_act = rf_child3.transform_vectors(
            np.tile(p, (10, 1)), rf_child2, timestamps=np.arange(10))
        v0t = rf_child3.transform_points((0., 0., 0.), rf_child2)
        vt = np.tile(pt, (5, 1)) - np.array(v0t)
        np.testing.assert_allclose(vt_act, vt, rtol=1.)

    @pytest.mark.parametrize('o, ot, p, pt, rc1, rc2, tc1, tc2',
                             transform_test_grid())
    def test_transform_points(self, o, ot, p, pt, rc1, rc2, tc1, tc2):
        """"""
        _, rf_child1, rf_child2 = get_rf_tree(tc1, rc1, tc2, rc2)
        rf_child3 = rbm.ReferenceFrame(
            'child3', rf_child1, timestamps=np.arange(5) + 2.5)

        # static reference frame + single vector
        pt_act = rf_child1.transform_points(p, rf_child2)
        np.testing.assert_allclose(pt_act, pt)

        # moving reference frame + single point
        pt_act = rf_child3.transform_points(p, rf_child2)
        np.testing.assert_allclose(pt_act, np.tile(pt, (5, 1)), rtol=1.)

        # moving reference frame + multiple vectors
        pt_act = rf_child3.transform_points(
            np.tile(p, (10, 1)), rf_child2, timestamps=np.arange(10))
        np.testing.assert_allclose(pt_act, np.tile(pt, (5, 1)), rtol=1.)

    @pytest.mark.parametrize('o, ot, p, pt, rc1, rc2, tc1, tc2',
                             transform_test_grid())
    def test_transform_quaternions(self, o, ot, p, pt, rc1, rc2, tc1, tc2):
        """"""
        _, rf_child1, rf_child2 = get_rf_tree(tc1, rc1, tc2, rc2)
        rf_child3 = rbm.ReferenceFrame(
            'child3', rf_child1, timestamps=np.arange(5) + 2.5)

        # static reference frame + single quaternion
        ot_act = rf_child1.transform_quaternions(o, rf_child2)
        npt.assert_allclose(np.abs(ot_act), np.abs(ot), rtol=1.)

        # moving reference frame + single quaternion
        ot_act = rf_child3.transform_quaternions(o, rf_child2)
        np.testing.assert_allclose(
            np.abs(ot_act), np.tile(np.abs(ot), (5, 1)), rtol=1.)

        # moving reference frame + multiple vectors
        ot_act = rf_child3.transform_quaternions(
            np.tile(o, (10, 1)), rf_child2, timestamps=np.arange(10))
        np.testing.assert_allclose(
            np.abs(ot_act), np.tile(np.abs(ot), (5, 1)), rtol=1.)
