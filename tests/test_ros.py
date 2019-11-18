import pytest
from .helpers import rf_test_grid, transform_test_grid, get_rf_tree

import numpy as np

from rigid_body_motion.ros import Transformer


class TestTransformer(object):

    def test_can_transform(self):
        """"""
        rf_world, _, _ = get_rf_tree()
        transformer = Transformer.from_reference_frame(rf_world)
        assert transformer.can_transform('child1', 'child2')

    @pytest.mark.parametrize('r, rc1, rc2, t, tc1, tc2', rf_test_grid())
    def test_lookup_transform(self, r, rc1, rc2, t, tc1, tc2):
        """"""
        rf_world, _, _ = get_rf_tree(tc1, rc1, tc2, rc2)
        transformer = Transformer.from_reference_frame(rf_world)
        t_act, r_act = transformer.lookup_transform('child2', 'child1')
        np.testing.assert_allclose(t_act, t)
        np.testing.assert_allclose(r_act, r)

    @pytest.mark.parametrize('o, ot, p, pt, rc1, rc2, tc1, tc2',
                             transform_test_grid())
    def test_transform_vector(self, o, ot, p, pt, rc1, rc2, tc1, tc2):
        """"""
        rf_world, _, _ = get_rf_tree(tc1, rc1, tc2, rc2)
        transformer = Transformer.from_reference_frame(rf_world)
        vt_act = transformer.transform_vector(p, 'child2', 'child1')
        v0t = transformer.transform_point((0., 0., 0.), 'child2', 'child1')
        vt = np.array(pt) - np.array(v0t)
        # large relative differences at machine precision
        np.testing.assert_allclose(vt_act, vt, rtol=1.)

    @pytest.mark.parametrize('o, ot, p, pt, rc1, rc2, tc1, tc2',
                             transform_test_grid())
    def test_transform_point(self, o, ot, p, pt, rc1, rc2, tc1, tc2):
        """"""
        rf_world, _, _ = get_rf_tree(tc1, rc1, tc2, rc2)
        transformer = Transformer.from_reference_frame(rf_world)
        pt_act = transformer.transform_point(p, 'child2', 'child1')
        np.testing.assert_allclose(pt_act, pt)

    @pytest.mark.parametrize('o, ot, p, pt, rc1, rc2, tc1, tc2',
                             transform_test_grid())
    def test_transform_quaternion(self, o, ot, p, pt, rc1, rc2, tc1, tc2):
        """"""
        rf_world, _, _ = get_rf_tree(tc1, rc1, tc2, rc2)
        transformer = Transformer.from_reference_frame(rf_world)
        ot_act = transformer.transform_quaternion(o, 'child2', 'child1')
        # large relative differences at machine precision
        np.testing.assert_allclose(ot_act, ot, rtol=1.)

    @pytest.mark.parametrize('o, ot, p, pt, rc1, rc2, tc1, tc2',
                             transform_test_grid())
    def test_transform_pose(self, o, ot, p, pt, rc1, rc2, tc1, tc2):
        """"""
        rf_world, _, _ = get_rf_tree(tc1, rc1, tc2, rc2)
        transformer = Transformer.from_reference_frame(rf_world)
        pt_act, ot_act = transformer.transform_pose(p, o, 'child2', 'child1')
        np.testing.assert_allclose(pt_act, pt)
        # large relative differences at machine precision
        np.testing.assert_allclose(ot_act, ot, rtol=1.)
