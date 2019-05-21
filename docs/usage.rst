=====
Usage
=====

.. code-block:: python

    import numpy as np
    import rigid_body_motion as rbm

    arr = np.ones((10, 2))
    rbm.transform(arr, outof='cartesian', into='polar')
