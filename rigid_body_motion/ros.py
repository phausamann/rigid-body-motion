""""""
try:
    import rospy
    import tf2_ros
    from geometry_msgs.msg import \
        TransformStamped, Vector3Stamped, Vector3, Point, Quaternion
except ImportError:
    # TODO setup environment for CI
    raise ImportError(
        'A ROS environment including rospy, tf2_ros and tf2_geometry_msgs is '
        'required for this module')

import numpy as np

from anytree import PreOrderIter

from rigid_body_motion.reference_frames import ReferenceFrame


def make_transform_msg(t, r, frame_id, child_frame_id, time=0.):
    """"""
    msg = TransformStamped()
    msg.header.stamp = rospy.Time.from_sec(time)
    msg.header.frame_id = frame_id
    msg.child_frame_id = child_frame_id
    msg.transform.translation = Vector3(*t)
    msg.transform.rotation = Quaternion(r[1], r[2], r[3], r[0])

    return msg


def static_rf_to_transform_msg(rf, time=0.):
    """"""
    return make_transform_msg(
        rf.translation, rf.rotation, rf.parent.name, rf.name, time=time)


def make_vector3_msg(v, frame_id, time=0.):
    """"""
    msg = Vector3Stamped()
    msg.header.stamp = rospy.Time.from_sec(time)
    msg.header.frame_id = frame_id
    msg.vector = Vector3(*v)

    return msg


class Transformer(object):

    def __init__(self, cache_time=None):
        """"""
        if cache_time is not None:
            self._buffer = tf2_ros.Buffer(
                cache_time=rospy.Duration.from_sec(cache_time), debug=False)
        else:
            self._buffer = tf2_ros.Buffer(debug=False)

    @staticmethod
    def from_reference_frame(reference_frame):
        """"""
        if len(reference_frame.ancestors) > 0:
            root = reference_frame.ancestors[0]
        else:
            root = reference_frame

        # get the first and last timestamps for all moving reference frames
        t_start_end = zip(*[
            (node.timestamps[0], node.timestamps[-1])
            for node in PreOrderIter(root) if hasattr(node, 'timestamps')
        ])

        if len(t_start_end) == 0:
            transformer = Transformer(cache_time=None)
        else:
            # cache time from earliest start to latest end
            cache_time = np.max(t_start_end[1]) - np.min(t_start_end[0])
            transformer = Transformer(cache_time=cache_time)

        for node in PreOrderIter(root):
            if isinstance(node, ReferenceFrame):
                if node.parent is not None:
                    transformer.set_transform_static(node)
            else:
                raise NotImplementedError()

        return transformer

    def set_transform_static(self, reference_frame):
        """"""
        self._buffer.set_transform_static(
            static_rf_to_transform_msg(reference_frame), 'default_authority')

    def can_transform(self, target_frame, source_frame, time=0.):
        """"""
        return self._buffer.can_transform(
            target_frame, source_frame, rospy.Time.from_sec(time))
