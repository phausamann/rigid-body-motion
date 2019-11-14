""""""
try:
    import rospy
    import tf2_ros
    import tf2_geometry_msgs
    from geometry_msgs.msg import \
        TransformStamped, Vector3Stamped, Vector3, PointStamped, Point, \
        QuaternionStamped, Quaternion
except ImportError:
    # try to import pyros_setup instead
    try:
        import pyros_setup
        pyros_setup.configurable_import().configure().activate()
        import rospy
        import tf2_ros
        import tf2_geometry_msgs
        from geometry_msgs.msg import \
            TransformStamped, Vector3Stamped, Vector3, PointStamped, Point, \
            QuaternionStamped, Quaternion
    except ImportError:
        raise ImportError(
            'A ROS environment including rospy, tf2_ros and tf2_geometry_msgs '
            'is required for this module')

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


def unpack_vector_msg(msg):
    """"""
    return msg.vector.x, msg.vector.y, msg.vector.z


def make_point_msg(p, frame_id, time=0.):
    """"""
    msg = PointStamped()
    msg.header.stamp = rospy.Time.from_sec(time)
    msg.header.frame_id = frame_id
    msg.point = Point(*p)

    return msg


def unpack_point_msg(msg):
    """"""
    return msg.point.x, msg.point.y, msg.point.z


def make_quaternion_msg(q, frame_id, time=0.):
    """"""
    msg = QuaternionStamped()
    msg.header.stamp = rospy.Time.from_sec(time)
    msg.header.frame_id = frame_id
    msg.quaternion = Quaternion(q[1], q[2], q[3], q[0])

    return msg


def unpack_quaternion_msg(msg):
    """"""
    return \
        msg.quaternion.w, msg.quaternion.x, msg.quaternion.y, msg.quaternion.z


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

    def transform_vector(self, vector, target_frame, source_frame, time=0.):
        """"""
        transform = self._buffer.lookup_transform(
            target_frame, source_frame, rospy.Time.from_sec(time))
        v_msg = make_vector3_msg(vector, source_frame, time)
        vt_msg = tf2_geometry_msgs.do_transform_vector3(v_msg, transform)

        return unpack_vector_msg(vt_msg)

    def transform_point(self, point, target_frame, source_frame, time=0.):
        """"""
        transform = self._buffer.lookup_transform(
            target_frame, source_frame, rospy.Time.from_sec(time))
        p_msg = make_point_msg(point, source_frame, time)
        pt_msg = tf2_geometry_msgs.do_transform_point(p_msg, transform)

        return unpack_point_msg(pt_msg)
