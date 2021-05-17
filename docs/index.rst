=================
rigid-body-motion
=================

Python utilities for estimating and transforming rigid body motion.


Hosted on GitHub: https://github.com/phausamann/rigid-body-motion


Overview
--------

This package provides a high-level interface for transforming arrays
describing motion of rigid bodies between different coordinate systems and
reference frames. The core of the reference frame handling is a fast
re-implementation of ROS's ``tf2`` library using ``numpy`` and
``numpy-quaternion``. The package also provides first-class support for
xarray_ data types.

.. _xarray: https://xarray.pydata.org


.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation

.. toctree::
   :maxdepth: 1
   :caption: User guide

   reference_frames
   velocities
   xarray
   ros

.. toctree::
   :maxdepth: 1
   :caption: Help & reference

   api
   roadmap
   contributing
   authors
   history


Indices and tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
