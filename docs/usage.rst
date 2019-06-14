=====
Usage
=====

Transform numpy arrays between coordinate systems:

.. code-block:: python

    import numpy as np
    import rigid_body_motion as rbm

    arr_cart = np.ones((10, 2))
    arr_polar = rbm.transform(arr_cart, outof='cartesian', into='polar')


Transform numpy arrays across a tree of static reference frames:

.. code-block:: python

    import numpy as np
    import rigid_body_motion as rbm

    rbm.register_frame('world')
    rbm.register_frame('child', parent='world', translation=(1., 0., 0.))
    rbm.register_frame('child2', parent='world', translation=(-1., 0., 0.))

    arr_child = np.ones((10, 3))
    arr_child2 = rbm.transform(arr_child, outof='child', into='child2')
