import rigid_body_motion as rbm
from rigid_body_motion.ros import Transformer


class TestTransformer(object):

    def test_from_reference_frame(self):
        """"""
        rf_world = rbm.ReferenceFrame('world')
        rf_child = rbm.ReferenceFrame('child', parent=rf_world)
        rf_child2 = rbm.ReferenceFrame('child2', parent=rf_world)

        transformer = Transformer.from_reference_frame(rf_child)

        assert transformer.can_transform('child', 'child2')
