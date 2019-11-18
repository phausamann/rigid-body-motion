import numpy as np

import pytest
from numpy import testing as npt
from .helpers import rf_test_grid, transform_test_grid, get_rf_tree

import rigid_body_motion as rbm
from rigid_body_motion.reference_frames import _register, _deregister


class TestReferenceFrameRegistry(object):

    def test_register(self):
        """"""
        rf_world = rbm.ReferenceFrame('world')
        _register(rf_world)
        assert rbm._rf_registry['world'] is rf_world

        with pytest.raises(ValueError):
            _register(rf_world)

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

    def test_get_parent_transform_matrix(self):
        """"""
        rf_world = rbm.ReferenceFrame('world')
        rf_child = rbm.ReferenceFrame('child', parent=rf_world,
                                      translation=(1., 0., 0.))

        actual = rf_child._get_parent_transformation_matrix()
        expected = np.eye(4)
        expected[0, 3] = 1.
        npt.assert_equal(actual, expected)

        actual = rf_child._get_parent_transformation_matrix(inverse=True)
        expected = np.eye(4)
        expected[0, 3] = -1.
        npt.assert_equal(actual, expected)

    @pytest.mark.parametrize('r, rc1, rc2, t, tc1, tc2', rf_test_grid())
    def test_get_transformation(self, r, rc1, rc2, t, tc1, tc2):
        """"""
        rf_world, rf_child1, rf_child2 = get_rf_tree(tc1, rc1, tc2, rc2)

        # child1 to world
        t_act, r_act = rf_child1.get_transformation(rf_world)
        npt.assert_almost_equal(t_act, tc1)
        npt.assert_almost_equal(r_act, rc1)

        # child1 to child2
        t_act, r_act = rf_child1.get_transformation(rf_child2)
        npt.assert_almost_equal(t_act, t)
        npt.assert_almost_equal(r_act, r)

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
        vt_act = rf_child1.transform_vectors(p, rf_child2)
        v0t = rf_child1.transform_points((0., 0., 0.), rf_child2)
        vt = np.array(pt) - np.array(v0t)
        # large relative differences at machine precision
        np.testing.assert_allclose(vt_act, vt, rtol=1.)

    @pytest.mark.parametrize('o, ot, p, pt, rc1, rc2, tc1, tc2',
                             transform_test_grid())
    def test_transform_points(self, o, ot, p, pt, rc1, rc2, tc1, tc2):
        """"""
        _, rf_child1, rf_child2 = get_rf_tree(tc1, rc1, tc2, rc2)
        pt_act = rf_child1.transform_points(p, rf_child2)
        np.testing.assert_allclose(pt_act, pt)

    @pytest.mark.parametrize('o, ot, p, pt, rc1, rc2, tc1, tc2',
                             transform_test_grid())
    def test_transform_quaternions(self, o, ot, p, pt, rc1, rc2, tc1, tc2):
        """"""
        _, rf_child1, rf_child2 = get_rf_tree(tc1, rc1, tc2, rc2)
        ot_act = rf_child1.transform_quaternions(o, rf_child2)
        # large relative differences at machine precision
        npt.assert_allclose(np.abs(ot_act), np.abs(ot), rtol=1.)
