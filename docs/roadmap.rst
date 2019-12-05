===================
Development roadmap
===================

The vision for this project is to provide a universal library for analysis
of recorded motion. Some of the categories where a lot of features are still
to be implemented are detailed below.


IO functions
------------

Import routines for a variety of data formats from common motion capture
and IMU systems.

Estimators
----------

Common transform estimators such as iterative closest points (ICP).

Pandas support
--------------

The same metadata-aware mechanism for pandas as for xarray.

ROS integration
---------------

Leverage tools provided by ROS to supplement functionality of the library.
One example would be 3D plotting with RViz. ROS has recently made its way to
`conda forge`_, and packages such as `jupyter-ros`_ facilitate integration
with modern Python workflows.

.. _conda forge: https://medium.com/@wolfv/ros-on-conda-forge-dca6827ac4b6
.. _jupyter-ros: https://github.com/RoboStack/jupyter-ros

Units
-----

Support for unit conversions and unit-aware transforms. Possibly leveraging
packages such as `pint`_ or `unit support in xarray`_.

.. _pint: https://github.com/hgrecco/pint
.. _unit support in xarray: https://github.com/pydata/xarray/issues/525
