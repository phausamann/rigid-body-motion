""""""
import os
import traceback

debug = os.environ.get("RBM_ROS_DEBUG")


class FailedImportStub:
    """ Class that raises import error on construction. """

    msg = "Reason: unknown"

    def __init__(self, *args, **kwargs):
        raise ImportError(
            f"Failed to import {type(self).__name__}.\n\n{self.msg}"
        )


try:
    from .transformer import (  # noqa
        ReferenceFrameTransformBroadcaster,
        Transformer,
    )
except ImportError:
    if debug:
        raise
    else:

        class ReferenceFrameTransformBroadcaster(FailedImportStub):
            msg = traceback.format_exc()

        class Transformer(FailedImportStub):
            msg = traceback.format_exc()


try:
    from .visualization import ReferenceFrameMarkerPublisher  # noqa
except ImportError:
    if debug:
        raise
    else:

        class ReferenceFrameMarkerPublisher(FailedImportStub):
            msg = traceback.format_exc()


try:
    from .io import RosbagReader, RosbagWriter  # noqa
except ImportError:
    if debug:
        raise
    else:

        class RosbagReader(FailedImportStub):
            msg = traceback.format_exc()

        class RosbagWriter(FailedImportStub):
            msg = traceback.format_exc()


from .utils import play_publisher  # noqa
