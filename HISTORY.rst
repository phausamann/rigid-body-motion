=======
History
=======

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
