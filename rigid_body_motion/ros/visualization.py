from io import BytesIO
from threading import Thread

import rospy
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from rigid_body_motion.core import _resolve_rf


def hex_to_rgba(h):
    """ Convert hex color string to ColorRGBA message.

    Parameters
    ----------
    h : str
        Hex color string in the format #RRGGBBAA.

    Returns
    -------
    c : ColorRGBA
        ColorRGBA message.
    """
    h = h.lstrip("#")
    return ColorRGBA(*(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4, 6)))


def get_marker(
    marker_type=Marker.LINE_STRIP,
    frame_id="world",
    scale=1.0,
    color="#ffffffff",
    position=(0.0, 0.0, 0.0),
    orientation=(0.0, 0.0, 0.0, 1.0),
):
    """ Create a Marker visualization message.

    Parameters
    ----------
    marker_type : int, default Marker.LINE_STRIP
        Type of the marker.

    frame_id : str, default "world"
        Name of the reference frame of the marker.

    scale : float or iterable of float, len 3, default 1.0
        Scale of the marker.

    color : str, default "#ffffffff"
        Color of the marker.

    position : iterable, len 3, default (0.0, 0.0, 0.0)
        Position of the marker wrt its reference frame.

    orientation : iterable, len 4, default (0.0, 0.0, 0.0, 1.0)
        Orientation of the marker wrt its reference frame.

    Returns
    -------
    marker: Marker
        Marker message.
    """
    marker = Marker()

    marker.type = marker_type
    marker.header.frame_id = frame_id

    if isinstance(scale, float):
        marker.scale = Vector3(scale, scale, scale)
    else:
        marker.scale = Vector3(*scale)

    marker.color = hex_to_rgba(color)
    marker.pose.orientation = Quaternion(*orientation)
    marker.pose.position = Point(*position)

    # TOD0: make configurable via vector
    marker.points = []

    return marker


class BaseMarkerPublisher:
    """ Base class for Marker publishers. """

    def __init__(self, marker, topic, publish_interval=0.0, verbose=False):
        """ Constructor.

        Parameters
        ----------
        marker : Marker
            Marker message to publish.

        topic : str
            Name of the topic on which to publish.

        publish_interval : float, default 0.0
            Time in seconds between publishing when calling ``spin``.
        """
        self.marker = marker
        self.topic = topic
        self.publish_interval = publish_interval

        self.last_message = BytesIO()

        self.publisher = rospy.Publisher(
            self.topic, Marker, queue_size=1, latch=True
        )
        if verbose:
            rospy.loginfo("Created marker publisher")

        self.stopped = False
        self._thread = None

    def publish(self):
        """ Publish a marker message. """
        current_message = BytesIO()
        self.marker.serialize(current_message)

        if current_message.getvalue() != self.last_message.getvalue():
            self.last_message = current_message
            self.publisher.publish(self.marker)

    def _spin_blocking(self):
        """ Continuously publish messages. """
        self.stopped = False

        while not rospy.is_shutdown() and not self.stopped:
            self.publish()
            rospy.sleep(self.publish_interval)

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
            self._spin_blocking()
        else:
            self._thread = Thread(target=self._spin_blocking)
            self._thread.start()

    def stop(self):
        """ Stop publishing. """
        self.stopped = True


class ReferenceFrameMarkerPublisher(BaseMarkerPublisher):
    """ Publisher for the translation of a reference frame wrt another. """

    def __init__(
        self,
        frame,
        base=None,
        topic=None,
        max_points=1000,
        publish_interval=0.0,
        scale=0.1,
        color="#ffffffff",
        verbose=False,
    ):
        """ Constructor.

        Parameters
        ----------
        frame : str or ReferenceFrame
            Reference frame for which to publish the translation.

        base : str or ReferenceFrame, optional
            Base reference wrt to which the translation is published. Defaults
            to the parent reference frame.

        topic : str, optional
            Name of the topic on which to publish. Defaults to "/<frame>/path".

        max_points : int, default 1000
            Maximum number of points to add to the marker. Actual translation
            array will be sub-sampled to this number of points.

        publish_interval : float, default 0.0
            Time in seconds between publishing when calling ``spin``.
        """
        self.frame = _resolve_rf(frame)
        self.base = _resolve_rf(base or self.frame.parent)
        self.translation, _, _ = self.frame.lookup_transform(self.base)

        marker = get_marker(
            frame_id=self.base.name, scale=(scale, 0.0, 0.0), color=color
        )
        show_every = self.translation.shape[0] // max_points
        marker.points = [Point(*row) for row in self.translation[::show_every]]

        topic = topic or f"/{self.frame.name}/path"

        super().__init__(
            marker, topic, publish_interval=publish_interval, verbose=verbose
        )

    def get_ros3d_widget(self, ros=None, tf_client=None):
        """ Get a ros3d.Marker widget to display in a ros3d.Viewer.

        Parameters
        ----------
        ros : jupyros.ros3d.ROSConnection, optional
            ros3d ROS connection instance.

        tf_client : jupyros.ros3d.TFClient, optional
            ros3d TF client instance.

        Returns
        -------
        jupyros.ros3d.Marker
            ros3d marker widget.
        """
        from jupyros import ros3d

        ros = ros or ros3d.ROSConnection()
        tf_client = tf_client or ros3d.TFClient(
            ros=ros, fixed_frame=self.base.name
        )

        return ros3d.Marker(
            ros=ros, tf_client=tf_client, topic=self.topic, path=""
        )
