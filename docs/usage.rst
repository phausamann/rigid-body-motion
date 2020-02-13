=====
Usage
=====

Coordinate system transforms
----------------------------

Transform numpy arrays between coordinate systems:

.. doctest::

    >>> import numpy as np
    >>> import rigid_body_motion as rbm
    >>> arr = [1., 1.]
    >>> rbm.transform_coordinates(arr, outof='cartesian', into='polar')
    array([1.41421356, 0.78539816])


Reference frame transforms
--------------------------

Transform numpy arrays across a tree of reference frames:

.. code-block:: python

    import numpy as np
    import rigid_body_motion as rbm

    rbm.register_frame('world')
    rbm.register_frame('child', parent='world', translation=(1., 0., 0.))
    rbm.register_frame('child2', parent='world', translation=(-1., 0., 0.))

    arr_child = np.ones((10, 3))
    arr_child2 = rbm.transform_points(
        arr_child, outof='child', into='child2')
