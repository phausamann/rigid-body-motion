from threading import Thread

import numpy as np
import pandas as pd
import rospy
from anytree import PreOrderIter
from geometry_msgs.msg import PoseStamped, TwistStamped

try:
    import rospkg
    import tf2_geometry_msgs
    import tf2_ros
    from tf.msg import tfMessage
except rospkg.ResourceNotFound:
    raise ImportError(
        "The rospkg module was found but tf2_ros failed to import, "
        "make sure you've set up the necessary environment variables"
    )

from rigid_body_motion.core import _resolve_rf
from rigid_body_motion.reference_frames import ReferenceFrame

from .msg import (
    make_point_msg,
    make_pose_msg,
    make_transform_msg,
    make_twist_msg,
    make_vector_msg,
    static_rf_to_transform_msg,
    unpack_point_msg,
    unpack_pose_msg,
    unpack_transform_msg,
    unpack_vector_msg,
)


class Transformer:
    def __init__(self, cache_time=None):
        """ Constructor.

        Parameters
        ----------
        cache_time : float, optional
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
                    (
                        node.timestamps[0].astype(float) / 1e9,
                        node.timestamps[-1].astype(float) / 1e9,
                    )
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
                    if node.timestamps is None:
                        transformer.set_transform_static(node)
                    else:
                        transformer.set_transforms(node)
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

    def set_transforms(self, reference_frame):
        """ Add transforms from moving reference frame to buffer.

        Parameters
        ----------
        reference_frame : ReferenceFrame
            Static reference frame to add.
        """
        for translation, rotation, timestamp in zip(
            reference_frame.translation,
            reference_frame.rotation,
            reference_frame.timestamps,
        ):
            self._buffer.set_transform(
                make_transform_msg(
                    translation,
                    rotation,
                    reference_frame.parent.name,
                    reference_frame.name,
                    timestamp.astype(float) / 1e9,
                ),
                "default_authority",
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

        return unpack_transform_msg(transform, stamped=True)

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

        return unpack_vector_msg(vt_msg, stamped=True)

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

        return unpack_point_msg(pt_msg, stamped=True)

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

        return unpack_pose_msg(pt_msg, stamped=True)[1]

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

        return unpack_pose_msg(pt_msg, stamped=True)


class ReferenceFrameTransformBroadcaster:
    """ TF broadcaster for the transform of a reference frame wrt another. """

    def __init__(
        self,
        frame,
        base=None,
        publish_pose=False,
        publish_twist=False,
        subscribe=False,
        twist_cutoff=None,
        twist_outlier_thresh=None,
    ):
        """ Constructor.

        Parameters
        ----------
        frame : str or ReferenceFrame
            Reference frame for which to publish the transform.

        base : str or ReferenceFrame, optional
            Base reference wrt to which the transform is published. Defaults
            to the parent reference frame.

        publish_pose : bool, default False
            If True, also publish a PoseStamped message on the topic
            "/<frame>/pose".

        publish_twist : bool, default False
            If True, also publish a TwistStamped message with the linear and
            angular velocity of the frame wrt the base on the topic
            "/<frame>/twist".

        subscribe : bool or str, default False
            If True, subscribe to the "/tf" topic and publish transforms
            when messages are broadcast whose `child_frame_id` is the name of
            the base frame. If the frame is a moving reference frame, the
            transform whose timestamp is the closest to the broadcast timestamp
            is published. `subscribe` can also be a string, in which case this
            broadcaster will be listening for TF messages with this
            `child_frame_id`.
        """
        self.frame = _resolve_rf(frame)
        self.base = _resolve_rf(base or self.frame.parent)
        (
            self.translation,
            self.rotation,
            self.timestamps,
        ) = self.frame.lookup_transform(self.base)

        if self.timestamps is None:
            self.broadcaster = tf2_ros.StaticTransformBroadcaster()
        else:
            self.timestamps = pd.Index(self.timestamps)
            self.broadcaster = tf2_ros.TransformBroadcaster()

        if publish_pose:
            self.pose_publisher = rospy.Publisher(
                f"/{self.frame.name}/pose",
                PoseStamped,
                queue_size=1,
                latch=True,
            )
        else:
            self.pose_publisher = None

        if publish_twist:
            self.linear, self.angular = self.frame.lookup_twist(
                reference=base,
                represent_in=self.frame,
                cutoff=twist_cutoff,
                outlier_thresh=twist_outlier_thresh,
            )
            self.twist_publisher = rospy.Publisher(
                f"/{self.frame.name}/twist",
                TwistStamped,
                queue_size=1,
                latch=True,
            )
        else:
            self.twist_publisher = None

        if subscribe:
            self.subscriber = rospy.Subscriber(
                "/tf", tfMessage, self.handle_incoming_msg
            )
            if isinstance(subscribe, str):
                self.subscribe_to_frame = subscribe
            else:
                self.subscribe_to_frame = self.base.name
        else:
            self.subscriber = None

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
            if self.pose_publisher is not None:
                pose = make_pose_msg(
                    self.translation, self.rotation, self.base.name,
                )
        else:
            idx = idx or self.idx
            ts = self.timestamps.values[idx].astype(float) / 1e9
            transform = make_transform_msg(
                self.translation[idx],
                self.rotation[idx],
                self.base.name,
                self.frame.name,
                ts,
            )
            if self.pose_publisher is not None:
                pose = make_pose_msg(
                    self.translation[idx],
                    self.rotation[idx],
                    self.base.name,
                    ts,
                )

        self.broadcaster.sendTransform(transform)

        if self.pose_publisher is not None:
            self.pose_publisher.publish(pose)

        if self.twist_publisher is not None:
            self.twist_publisher.publish(
                make_twist_msg(
                    self.linear[idx], self.angular[idx], self.frame.name
                )
            )

    def handle_incoming_msg(self, msg):
        """ Publish on incoming message. """
        for transform in msg.transforms:
            if transform.child_frame_id == self.subscribe_to_frame:
                if self.timestamps is not None:
                    ts = pd.to_datetime(
                        rospy.Time.to_sec(transform.header.stamp), unit="s"
                    )
                    idx = self.timestamps.get_loc(ts, method="nearest")
                    self.publish(idx)
                else:
                    self.publish()

    def _spin_blocking(self):
        """ Continuously publish messages. """
        self.stopped = False

        if self.subscriber is None and self.timestamps is not None:
            while not rospy.is_shutdown() and not self.stopped:
                self.publish()
                self.idx = (self.idx + 1) % len(self.timestamps)
                if isinstance(self.timestamps, pd.DatetimeIndex):
                    dt = (
                        self.timestamps.values[self.idx].astype(float) / 1e9
                        - self.timestamps.values[self.idx - 1].astype(float)
                        / 1e9
                        if self.idx > 0
                        else 0.0
                    )
                else:
                    dt = float(
                        self.timestamps.values[self.idx]
                        - self.timestamps.values[self.idx - 1]
                        if self.idx > 0
                        else 0.0
                    )

                rospy.sleep(dt)
        else:
            rospy.spin()

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
        if self.timestamps is None:
            self.publish()
        elif block:
            self._spin_blocking()
        else:
            self._thread = Thread(target=self._spin_blocking)
            self._thread.start()

    def stop(self):
        """ Stop publishing. """
        self.stopped = True
