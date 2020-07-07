from threading import Thread

import numpy as np
from anytree import PreOrderIter

import PyKDL
import rospy
from geometry_msgs.msg import (
    PointStamped,
    Vector3Stamped,
    PoseStamped,
    TransformStamped,
)

from .msg import static_rf_to_transform_msg

try:
    import rospkg
    import tf2_ros
except rospkg.ResourceNotFound:
    raise ImportError(
        "The rospkg module was found but tf2_ros failed to import, "
        "make sure you've set up the necessary environment variables"
    )

from rigid_body_motion.core import _resolve_rf
from rigid_body_motion.reference_frames import ReferenceFrame
from .msg import (
    make_transform_msg,
    unpack_transform_msg,
    make_vector_msg,
    unpack_vector_msg,
    make_point_msg,
    unpack_point_msg,
    make_pose_msg,
    unpack_pose_msg,
)


class tf2_geometry_msgs:
    """ Copied routines from tf2_geometry_msgs. """

    @classmethod
    def transform_to_kdl(cls, t):
        return PyKDL.Frame(
            PyKDL.Rotation.Quaternion(
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w,
            ),
            PyKDL.Vector(
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z,
            ),
        )

    @classmethod
    def do_transform_point(cls, point, transform):
        p = cls.transform_to_kdl(transform) * PyKDL.Vector(
            point.point.x, point.point.y, point.point.z
        )
        res = PointStamped()
        res.point.x = p[0]
        res.point.y = p[1]
        res.point.z = p[2]
        res.header = transform.header
        return res

    @classmethod
    def do_transform_vector3(cls, vector3, transform):
        transform.transform.translation.x = 0
        transform.transform.translation.y = 0
        transform.transform.translation.z = 0
        p = cls.transform_to_kdl(transform) * PyKDL.Vector(
            vector3.vector.x, vector3.vector.y, vector3.vector.z
        )
        res = Vector3Stamped()
        res.vector.x = p[0]
        res.vector.y = p[1]
        res.vector.z = p[2]
        res.header = transform.header
        return res

    @classmethod
    def do_transform_pose(cls, pose, transform):
        f = cls.transform_to_kdl(transform) * PyKDL.Frame(
            PyKDL.Rotation.Quaternion(
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            ),
            PyKDL.Vector(
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
            ),
        )
        res = PoseStamped()
        res.pose.position.x = f.p[0]
        res.pose.position.y = f.p[1]
        res.pose.position.z = f.p[2]
        (
            res.pose.orientation.x,
            res.pose.orientation.y,
            res.pose.orientation.z,
            res.pose.orientation.w,
        ) = f.M.GetQuaternion()
        res.header = transform.header
        return res


class Transformer(object):
    def __init__(self, cache_time=None):
        """ Constructor.

        Parameters
        ----------
        cache_time : float
            Cache time of the buffer in seconds.
        """
        if cache_time is not None:
            self._buffer = tf2_ros.Buffer(
                cache_time=rospy.Duration.from_sec(cache_time), debug=False
            )
        else:
            self._buffer = tf2_ros.Buffer(debug=False)

    @staticmethod
    def from_reference_frame(reference_frame):
        """ Construct Transformer instance from static reference frame tree.

        Parameters
        ----------
        reference_frame : ReferenceFrame
            Reference frame instance from which to construct the transformer.

        Returns
        -------
        Transformer
            Transformer instance.
        """
        root = reference_frame.root

        # get the first and last timestamps for all moving reference frames
        t_start_end = list(
            zip(
                *[
                    (node.timestamps[0], node.timestamps[-1])
                    for node in PreOrderIter(root)
                    if node.timestamps is not None
                ]
            )
        )

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
        """ Add static transform from reference frame to buffer.

        Parameters
        ----------
        reference_frame : ReferenceFrame
            Static reference frame to add.
        """
        self._buffer.set_transform_static(
            static_rf_to_transform_msg(reference_frame), "default_authority"
        )

    def can_transform(self, target_frame, source_frame, time=0.0):
        """ Check if transform from source to target frame is possible.

        Parameters
        ----------
        target_frame : str
            Name of the frame to transform into.

        source_frame : str
            Name of the input frame.

        time : float, default 0.0
            Time at which to get the transform. (0 will get the latest)

        Returns
        -------
        bool
            True if the transform is possible, false otherwise.
        """
        return self._buffer.can_transform(
            target_frame, source_frame, rospy.Time.from_sec(time)
        )

    def lookup_transform(self, target_frame, source_frame, time=0.0):
        """ Get the transform from the source frame to the target frame.

        Parameters
        ----------
        target_frame : str
            Name of the frame to transform into.

        source_frame : str
            Name of the input frame.

        time : float, default 0.0
            Time at which to get the transform. (0 will get the latest)

        Returns
        -------
        t : tuple, len 3
            The translation between the frames.

        r : tuple, len 4
            The rotation between the frames.
        """
        transform = self._buffer.lookup_transform(
            target_frame, source_frame, rospy.Time.from_sec(time)
        )

        return unpack_transform_msg(transform)

    def transform_vector(self, v, target_frame, source_frame, time=0.0):
        """ Transform a vector from the source frame to the target frame.

        Parameters
        ----------
        v : iterable, len 3
            Input vector in source frame.

        target_frame : str
            Name of the frame to transform into.

        source_frame : str
            Name of the input frame.

        time : float, default 0.0
            Time at which to get the transform. (0 will get the latest)

        Returns
        -------
        tuple, len 3
            Transformed vector in target frame.
        """
        transform = self._buffer.lookup_transform(
            target_frame, source_frame, rospy.Time.from_sec(time)
        )
        v_msg = make_vector_msg(v, source_frame, time)
        vt_msg = tf2_geometry_msgs.do_transform_vector3(v_msg, transform)

        return unpack_vector_msg(vt_msg)

    def transform_point(self, p, target_frame, source_frame, time=0.0):
        """ Transform a point from the source frame to the target frame.

        Parameters
        ----------
        p : iterable, len 3
            Input point in source frame.

        target_frame : str
            Name of the frame to transform into.

        source_frame : str
            Name of the input frame.

        time : float, default 0.0
            Time at which to get the transform. (0 will get the latest)

        Returns
        -------
        tuple, len 3
            Transformed point in target frame.
        """
        transform = self._buffer.lookup_transform(
            target_frame, source_frame, rospy.Time.from_sec(time)
        )
        p_msg = make_point_msg(p, source_frame, time)
        pt_msg = tf2_geometry_msgs.do_transform_point(p_msg, transform)

        return unpack_point_msg(pt_msg)

    def transform_quaternion(self, q, target_frame, source_frame, time=0.0):
        """ Transform a quaternion from the source frame to the target frame.

        Parameters
        ----------
        q : iterable, len 4
            Input quaternion in source frame.

        target_frame : str
            Name of the frame to transform into.

        source_frame : str
            Name of the input frame.

        time : float, default 0.0
            Time at which to get the transform. (0 will get the latest)

        Returns
        -------
        tuple, len 4
            Transformed quaternion in target frame.
        """
        transform = self._buffer.lookup_transform(
            target_frame, source_frame, rospy.Time.from_sec(time)
        )
        p_msg = make_pose_msg((0.0, 0.0, 0.0), q, source_frame, time)
        pt_msg = tf2_geometry_msgs.do_transform_pose(p_msg, transform)

        return unpack_pose_msg(pt_msg)[1]

    def transform_pose(self, p, o, target_frame, source_frame, time=0.0):
        """ Transform a pose from the source frame to the target frame.

        Parameters
        ----------
        p : iterable, len 3
            Input position in source frame.

        o : iterable, len 3
            Input orientation in source frame.

        target_frame : str
            Name of the frame to transform into.

        source_frame : str
            Name of the input frame.

        time : float, default 0.0
            Time at which to get the transform. (0 will get the latest)

        Returns
        -------
        pt : tuple, len 3
            Transformed position in target frame.

        ot : tuple, len 4
            Transformed orientation in target frame.
        """
        transform = self._buffer.lookup_transform(
            target_frame, source_frame, rospy.Time.from_sec(time)
        )
        p_msg = make_pose_msg(p, o, source_frame, time)
        pt_msg = tf2_geometry_msgs.do_transform_pose(p_msg, transform)

        return unpack_pose_msg(pt_msg)


class ReferenceFrameTransformBroadcaster:
    """ TF broadcaster for the transform of a reference frame wrt another. """

    def __init__(
        self,
        frame,
        base=None,
        subscribe_to=None,
        subscriber_msg_type=TransformStamped,
    ):
        """ Constructor.

        Parameters
        ----------
        frame : str or ReferenceFrame
            Reference frame for which to publish the transform.

        base : str or ReferenceFrame, optional
            Base reference wrt to which the transform is published. Defaults
            to the parent reference frame.
        """
        self.frame = _resolve_rf(frame)
        self.base = _resolve_rf(base or self.frame.parent)
        (
            self.translation,
            self.rotation,
            self.timestamps,
        ) = self.frame.get_transformation(self.base)

        if self.timestamps is None:
            self.broadcaster = tf2_ros.TransformBroadcaster()
        else:
            self.broadcaster = tf2_ros.StaticTransformBroadcaster()

        if subscribe_to is not None:
            self.subscriber = rospy.Subscriber(
                subscribe_to, subscriber_msg_type
            )

        self.idx = 0
        self.stopped = False
        self._thread = None

    def publish(self, idx=None):
        """ Publish a transform message.

        Parameters
        ----------
        idx : int, optional
            Index of the transform to publish for a moving reference frame.
            Uses ``self.idx`` as default.
        """
        if self.timestamps is None:
            transform = make_transform_msg(
                self.translation,
                self.rotation,
                self.base.name,
                self.frame.name,
            )
        else:
            transform = make_transform_msg(
                self.translation[idx or self.idx],
                self.rotation[idx or self.idx],
                self.base.name,
                self.frame.name,
                self.timestamps[idx or self.idx].astype(float) / 1e9,
            )

        self.broadcaster.sendTransform(transform)

    def handle_incoming_msg(self, msg):
        """ Publish on incoming message. """

    def _spin_blocking(self):
        """ Continuously publish messages. """
        self.stopped = False

        while not rospy.is_shutdown() and not self.stopped:
            self.publish()
            self.idx = (self.idx + 1) % len(self.timestamps)
            dt = (
                self.timestamps[self.idx].astype(float) / 1e9
                - self.timestamps[self.idx - 1].astype(float) / 1e9
                if self.idx > 0
                else 0.0
            )
            rospy.sleep(dt)

        self.stopped = True

    def spin(self, block=False):
        """ Continuously publish messages.

        Parameters
        ----------
        block: bool, default False
            If True, this method will block until the publisher is stopped,
            e.g. by calling stop(). Otherwise, the main loop is
            dispatched to a separate thread which is returned by this
            function.

        Returns
        -------
        thread: threading.Thread
            If `block=True`, the Thread instance that runs the loop.
        """
        if block:
            return self._spin_blocking()
        else:
            self._thread = Thread(target=self._spin_blocking)
            self._thread.start()
            return self._thread

    def stop(self):
        """ Stop publishing. """
        self.stopped = True
