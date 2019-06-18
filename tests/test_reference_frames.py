import os

import numpy as np
import pandas as pd

import pytest
from numpy import testing as npt

from quaternion import from_euler_angles, as_float_array

import rigid_body_motion as rbm
from rigid_body_motion.reference_frames import _register, _deregister

test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')


def mock_quaternion(*angles):
    """"""
    return as_float_array(from_euler_angles(*angles))


def load_csv(filepath):
    """"""
    df = pd.read_csv(filepath, header=[0, 1], index_col=0)
    l = [[tuple(r) for r in df[c].values] for c in df.columns.levels[0]]
    return list(zip(*l))


rf_test_grid = load_csv(os.path.join(test_data_dir, 'rf_test_grid.csv'))


class TestReferenceFrameRegistry(object):
    """"""

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
    """"""

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

    @pytest.mark.parametrize('r, rc1, rc2, t, tc1, tc2', rf_test_grid)
    def test_get_transformation(self, r, rc1, rc2, t, tc1, tc2):
        """"""
        rf_world = rbm.ReferenceFrame('world')

        rf_child1 = rbm.ReferenceFrame(
            'child1', parent=rf_world, translation=tc1, rotation=rc1)
        rf_child2 = rbm.ReferenceFrame(
            'child2', parent=rf_world, translation=tc2, rotation=rc2)

        t_act, r_act = rf_child1.get_transformation(rf_child2)
        npt.assert_almost_equal(t_act, t)
        npt.assert_almost_equal(r_act, r)
