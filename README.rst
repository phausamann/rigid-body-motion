.. image:: https://img.shields.io/travis/com/phausamann/rigid-body-motion.svg
        :target: https://travis-ci.com/phausamann/rigid-body-motion

.. image:: https://readthedocs.org/projects/rigid-body-motion/badge/?version=latest
        :target: https://rigid-body-motion.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/rigid-body-motion.svg
        :target: https://pypi.python.org/pypi/rigid-body-motion



=================
rigid-body-motion
=================

Python utilities for estimating and transforming rigid body motion.


Documentation: https://rigid-body-motion.readthedocs.io


Features
--------

* Transform numpy arrays describing rigid body motion between different
  coordinate systems and reference frames


Installation
------------

.. code-block:: console

    $ pip install git+https://github.com/phausamann/rigid-body-motion.git


Examples
--------

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


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
