.. image:: https://github.com/phausamann/rigid-body-motion/actions/workflows/build.yml/badge.svg
        :target: https://github.com/phausamann/rigid-body-motion/actions/workflows/build.yml

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

Highlights
----------

rigid-body-motion makes it possible to:

* Construct trees of static and moving reference frames
* Lookup transforms and velocities across the tree
* Seamlessly transform positions, orientations and velocities across the tree
* Estimate transforms from motion data
* Transform data into different coordinate representations
* Import data from common motion tracking systems
* Visualize reference frames and motion data with matplotlib or RViz
* ... and more!

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
