import pytest

from .helpers import rf_test_grid, transform_test_grid, get_rf_tree

import numpy as np


@pytest.fixture()
def Transformer():
    """"""
    tf = pytest.importorskip("rigid_body_motion.ros.transformer")
    return tf.Transformer


class TestTransformer(object):
    def test_can_transform(self, Transformer):
        """"""
        rf_world, _, _ = get_rf_tree()
        transformer = Transformer.from_reference_frame(rf_world)
        assert transformer.can_transform("child1", "child2")

    @pytest.mark.parametrize("r, rc1, rc2, t, tc1, tc2", rf_test_grid())
    def test_lookup_transform(self, r, rc1, rc2, t, tc1, tc2, Transformer):
        """"""
        rf_world, _, _ = get_rf_tree(tc1, rc1, tc2, rc2)
        transformer = Transformer.from_reference_frame(rf_world)
        t_act, r_act = transformer.lookup_transform("child2", "child1")
        np.testing.assert_allclose(t_act, t)
        np.testing.assert_allclose(r_act, r)

    @pytest.mark.parametrize(
        "o, ot, p, pt, rc1, rc2, tc1, tc2", transform_test_grid()
    )
    def test_transform_vector(
        self, o, ot, p, pt, rc1, rc2, tc1, tc2, Transformer
    ):
        """"""
        rf_world, _, _ = get_rf_tree(tc1, rc1, tc2, rc2)
        transformer = Transformer.from_reference_frame(rf_world)
        vt_act = transformer.transform_vector(p, "child2", "child1")
        v0t = transformer.transform_point((0.0, 0.0, 0.0), "child2", "child1")
        vt = np.array(pt) - np.array(v0t)
        # large relative differences at machine precision
        np.testing.assert_allclose(vt_act, vt, rtol=1.0)

    @pytest.mark.parametrize(
        "o, ot, p, pt, rc1, rc2, tc1, tc2", transform_test_grid()
    )
    def test_transform_point(
        self, o, ot, p, pt, rc1, rc2, tc1, tc2, Transformer
    ):
        """"""
        rf_world, _, _ = get_rf_tree(tc1, rc1, tc2, rc2)
        transformer = Transformer.from_reference_frame(rf_world)
        pt_act = transformer.transform_point(p, "child2", "child1")
        np.testing.assert_allclose(pt_act, pt)

    @pytest.mark.parametrize(
        "o, ot, p, pt, rc1, rc2, tc1, tc2", transform_test_grid()
    )
    def test_transform_quaternion(
        self, o, ot, p, pt, rc1, rc2, tc1, tc2, Transformer
    ):
        """"""
        rf_world, _, _ = get_rf_tree(tc1, rc1, tc2, rc2)
        transformer = Transformer.from_reference_frame(rf_world)
        ot_act = transformer.transform_quaternion(o, "child2", "child1")
        # large relative differences at machine precision
        np.testing.assert_allclose(ot_act, ot, rtol=1.0)

    @pytest.mark.parametrize(
        "o, ot, p, pt, rc1, rc2, tc1, tc2", transform_test_grid()
    )
    def test_transform_pose(
        self, o, ot, p, pt, rc1, rc2, tc1, tc2, Transformer
    ):
        """"""
        rf_world, _, _ = get_rf_tree(tc1, rc1, tc2, rc2)
        transformer = Transformer.from_reference_frame(rf_world)
        pt_act, ot_act = transformer.transform_pose(p, o, "child2", "child1")
        np.testing.assert_allclose(pt_act, pt)
        # large relative differences at machine precision
        np.testing.assert_allclose(ot_act, ot, rtol=1.0)


class TestVisualization:
    def test_hex_to_rgba(self):
        """"""
        from rigid_body_motion.ros.visualization import hex_to_rgba

        color_msg = hex_to_rgba("#ffffffff")
        assert color_msg.r == 1.0
        assert color_msg.g == 1.0
        assert color_msg.b == 1.0
        assert color_msg.a == 1.0

    def test_get_marker(self):
        """"""
        from rigid_body_motion.ros.visualization import get_marker

        marker_msg = get_marker()
        assert marker_msg.type == 4
        assert marker_msg.header.frame_id == "world"
