=======
History
=======


0.9.3 (June 17th, 2025)
-----------------------

Dependency changes
~~~~~~~~~~~~~~~~~~

* Removed unnecessary ``<2`` restriction on ``numpy`` dependency.


0.9.2 (February 9th, 2025)
--------------------------

Bug fixes & improvements
~~~~~~~~~~~~~~~~~~~~~~~~

* Fixed an error related to a deprecated matplotlib.pyplot reference, see `Issue #35 <https://github.com/phausamann/rigid-body-motion/issues/35>`.

Internal changes
~~~~~~~~~~~~~~~~

* Packaging is now handled by ``uv``.


0.9.1 (January 13th, 2022)
--------------------------

Bug fixes & improvements
~~~~~~~~~~~~~~~~~~~~~~~~

* Fixed package installation through pip (version 0.9.0 is no longer available).


0.9.0 (December 29th, 2021)
---------------------------

Breaking changes
~~~~~~~~~~~~~~~~

* Dropped support for Python 3.6.

Bug fixes & improvements
~~~~~~~~~~~~~~~~~~~~~~~~

* Fixed issue with matplotlib versions >= 3.5.


0.8.0 (May 27th, 2021)
----------------------

New features
~~~~~~~~~~~~

* New ``ros.init_node`` method to initialize a ROS node and optionally start
  a ROS master.


Bug fixes & improvements
~~~~~~~~~~~~~~~~~~~~~~~~
* All ROS dependencies are now lazily imported.


0.7.0 (May 19th, 2021)
----------------------

New features
~~~~~~~~~~~~

* New ``from_euler_angles`` utility method.


Bug fixes & improvements
~~~~~~~~~~~~~~~~~~~~~~~~

* Importing ROS interface classes will not fail silently anymore and instead
  show the traceback of the import error.


0.6.0 (May 17th, 2021)
----------------------

Breaking changes
~~~~~~~~~~~~~~~~

* Example data is now fetched via the ``pooch`` library and no longer a part
  of the package itself.

New features
~~~~~~~~~~~~

* New ``io`` module for import/export methods.
* New ``ros.RosbagWriter`` class for writing rosbag files.


0.5.0 (March 16th, 2021)
------------------------

Breaking changes
~~~~~~~~~~~~~~~~

* Top-level reference frame transform and lookup methods now all accept a
  ``return_timestamps`` argument that is ``False`` by default. Previously,
  methods would return timestamps only if the result of the transformation was
  timestamped. This does not affect the xarray interface.
* ``lookup_transform`` now returns the correct transformation from the base
  frame to the target frame (instead of the other way around).
* ``ReferenceFrame.get_transformation`` is deprecated and replaced by
  ``ReferenceFrame.lookup_transform``.

New features
~~~~~~~~~~~~

* New ``plot`` module with plotting methods for static reference frames and
  arrays of points, quaternions and vectors.
* New ``lookup_pose`` method that calculates the pose of a frame wrt another.

Bug fixes & improvements
~~~~~~~~~~~~~~~~~~~~~~~~

* Fixed ``"reference_frame"`` attribute incorrectly set by
  ``transform_vectors``.


0.4.1 (February 18th, 2021)
---------------------------

Bug fixes & improvements
~~~~~~~~~~~~~~~~~~~~~~~~

* Fixed ``transform_coordinates`` failing when spatial dimension is first
  axis of array.
* Fixed ``transform_linear_velocity`` and ``transform_angular_velocity``
  failing when reference frame or moving frame is transformed across only
  static transforms.
* Added ``allow_static`` parameter to ``lookup_twist``,
  ``lookup_angular_velocity`` and ``lookup_linear_velocity`` to return zero
  velocity and no timestamps across only static transforms.


0.4.0 (February 11th, 2021)
---------------------------

New features
~~~~~~~~~~~~

* New ``lookup_linear_velocity`` and ``lookup_angular_velocity`` top-level
  methods.
* New ``render_tree`` top-level method for printing out a graphical
  representation of a reference frame tree.
* ``lookup_twist`` now accepts a ``mode`` parameter to specify the mode for
  angular velocity calculation.

Bug fixes & improvements
~~~~~~~~~~~~~~~~~~~~~~~~

* Fixed a bug where estimated angular velocity was all NaN when orientation
  contained NaNs.


0.3.0 (December 8th, 2020)
--------------------------

New features
~~~~~~~~~~~~

* Reference frames with timestamps now accept the ``discrete`` parameter,
  allowing for transformations to be fixed from their timestamp into the
  future.
* ``rbm`` accessor for DataArrays implementing ``qinterp`` and ``qinv``
  methods.
* New ``best_fit_rotation`` and ``qinterp`` top-level methods.

Bug fixes & improvements
~~~~~~~~~~~~~~~~~~~~~~~~

* Refactor of internal timestamp matching mechanism defining a clear priority
  for target timestamps. This can result in slight changes of timestamps
  and arrays returned by transformations but will generally produce more
  accurate results.
* Added ``mode`` and ``outlier_thresh`` arguments to
  ``estimate_angular_velocity``.
* Fixed issues with ``iterative_closest_point``.


0.2.0 (October 22nd, 2020)
--------------------------

New features
~~~~~~~~~~~~

* New ``estimate_linear_velocity`` and ``estimate_angular_velocity`` top-level
  methods.
* New ``qmul`` top-level method for multiplying quaternions.


0.1.2 (October 7th, 2020)
-------------------------

Improvements
~~~~~~~~~~~~

* Use SQUAD instead of linear interpolation for quaternions.


0.1.1 (September 17th, 2020)
----------------------------

Bug fixes
~~~~~~~~~

* Fix transformations failing for DataArrays with non-numeric coords.


0.1.0 (September 17th, 2020)
----------------------------

* First release
