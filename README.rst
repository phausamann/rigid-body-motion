.. image:: https://img.shields.io/pypi/v/rigid-body-motion.svg
        :target: https://pypi.python.org/pypi/rigid-body-motion

.. image:: https://img.shields.io/travis/com/phausamann/rigid-body-motion.svg
        :target: https://travis-ci.com/phausamann/rigid-body-motion

.. image:: https://readthedocs.org/projects/rigid-body-motion/badge/?version=latest
        :target: https://rigid-body-motion.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


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


Example
-------

.. code-block:: python

    import numpy as np
    import rigid_body_motion as rbm

    arr = np.ones((10, 2))
    rbm.transform(arr, outof='cartesian', into='polar')


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
