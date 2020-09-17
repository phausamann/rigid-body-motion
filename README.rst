.. image:: https://img.shields.io/travis/com/phausamann/rigid-body-motion.svg
        :target: https://travis-ci.com/phausamann/rigid-body-motion

.. image:: https://readthedocs.org/projects/rigid-body-motion/badge/?version=latest
        :target: https://rigid-body-motion.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/rigid-body-motion.svg
        :target: https://pypi.python.org/pypi/rigid-body-motion

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
        :target: https://github.com/psf/black


=================
rigid-body-motion
=================

Python utilities for estimating and transforming rigid body motion.


Documentation: https://rigid-body-motion.readthedocs.io


Overview
--------

This package provides a high-level interface for transforming arrays
describing motion of rigid bodies between different coordinate systems and
reference frames. The core of the reference frame handling is a fast
re-implementation of ROS's ``tf2`` library using ``numpy`` and
``numpy-quaternion``. The package also provides first-class support for
xarray_ data types.

.. _xarray: https://xarray.pydata.org

Installation
------------

rigid-body-motion can be installed via ``pip``:

.. code-block:: console

    $ pip install rigid-body-motion

or via ``conda``:

.. code-block:: console

    $ conda install -c phausamann -c conda-forge rigid-body-motion


Examples
--------

Transform numpy arrays between coordinate systems:

.. code-block:: python

    import numpy as np
    import rigid_body_motion as rbm

    arr_cart = np.ones((10, 2))
    arr_polar = rbm.transform_coordinates(arr_cart, outof="cartesian", into="polar")


Transform numpy arrays across a tree of reference frames:

.. code-block:: python

    import numpy as np
    import rigid_body_motion as rbm

    rbm.register_frame("world")
    rbm.register_frame("child", parent="world", translation=(1., 0., 0.))
    rbm.register_frame("child2", parent="world", translation=(-1., 0., 0.))

    arr_child = np.ones((10, 3))
    arr_child2 = rbm.transform_points(arr_child, outof="child", into="child2")


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
