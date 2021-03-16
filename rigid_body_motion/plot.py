def fn_stub(*args, **kwargs):
    """ Stub function that raises an error on use. """
    raise ImportError("Install matplotlib to use plotting functions")


try:
    from .plotting import (
        plot_points,
        plot_quaternions,
        plot_reference_frame,
        plot_vectors,
    )

    reference_frame = plot_reference_frame
    points = plot_points
    quaternions = plot_quaternions
    vectors = plot_vectors
except ImportError:
    reference_frame = fn_stub
    points = fn_stub
    quaternions = fn_stub
    vectors = fn_stub
