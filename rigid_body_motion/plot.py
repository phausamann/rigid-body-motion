def fn_stub(*args, **kwargs):
    """ Stub function that raises an error on use. """
    raise ImportError("Install matplotlib to use plotting functions")


try:
    from .plotting import plot_points, plot_reference_frame

    reference_frame = plot_reference_frame
    points = plot_points
except ImportError:
    reference_frame = fn_stub
    points = fn_stub
