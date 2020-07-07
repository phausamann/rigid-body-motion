import rospy
from geometry_msgs.msg import (
    Point,
    PointStamped,
    PoseStamped,
    Quaternion,
    QuaternionStamped,
    TransformStamped,
    Vector3,
    Vector3Stamped,
)


def make_transform_msg(t, r, frame_id, child_frame_id, time=0.0):
    """ Create a TransformStamped message. """
    msg = TransformStamped()
    msg.header.stamp = rospy.Time.from_sec(time)
    msg.header.frame_id = frame_id
    msg.child_frame_id = child_frame_id
    msg.transform.translation = Vector3(*t)
    msg.transform.rotation = Quaternion(r[1], r[2], r[3], r[0])

    return msg


def unpack_transform_msg(msg):
    """ Get translation and rotation from a TransformStamped message. """
    t = msg.transform.translation
    r = msg.transform.rotation
    return (t.x, t.y, t.z), (r.w, r.x, r.y, r.z)


def make_pose_msg(p, o, frame_id, time=0.0):
    """ Create a PoseStamped message. """
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.from_sec(time)
    msg.header.frame_id = frame_id
    msg.pose.position = Point(*p)
    msg.pose.orientation = Quaternion(o[1], o[2], o[3], o[0])

    return msg


def unpack_pose_msg(msg):
    """ Get position and orientation from a PoseStamped message. """
    p = msg.pose.position
    o = msg.pose.orientation
    return (p.x, p.y, p.z), (o.w, o.x, o.y, o.z)


def make_vector_msg(v, frame_id, time=0.0):
    """ Create a Vector3Stamped message. """
    msg = Vector3Stamped()
    msg.header.stamp = rospy.Time.from_sec(time)
    msg.header.frame_id = frame_id
    msg.vector = Vector3(*v)

    return msg


def unpack_vector_msg(msg):
    """ Get coordinates from a Vector3Stamped message. """
    v = msg.vector
    return v.x, v.y, v.z


def make_point_msg(p, frame_id, time=0.0):
    """ Create a PointStamped message. """
    msg = PointStamped()
    msg.header.stamp = rospy.Time.from_sec(time)
    msg.header.frame_id = frame_id
    msg.point = Point(*p)

    return msg


def unpack_point_msg(msg):
    """ Get coordinates from a PointStamped message. """
    p = msg.point
    return p.x, p.y, p.z


def make_quaternion_msg(q, frame_id, time=0.0):
    """ Create a QuaternionStamped message. """
    msg = QuaternionStamped()
    msg.header.stamp = rospy.Time.from_sec(time)
    msg.header.frame_id = frame_id
    msg.quaternion = Quaternion(q[1], q[2], q[3], q[0])

    return msg


def unpack_quaternion_msg(msg):
    """ Get coordinates from a QuaternionStamped message. """
    q = msg.quaternion
    return q.w, q.x, q.y, q.z


def static_rf_to_transform_msg(rf, time=0.0):
    """ Convert a static ReferenceFrame to a TransformStamped message.

    Parameters
    ----------
    rf : ReferenceFrame
        Static reference frame.

    time : float, default 0.0
        The time of the message.

    Returns
    -------
    msg : TransformStamped
        TransformStamped message.
    """
    return make_transform_msg(
        rf.translation, rf.rotation, rf.parent.name, rf.name, time=time
    )
