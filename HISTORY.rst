=======
History
=======

0.3.0 (Unreleased)
------------------

New features
~~~~~~~~~~~~

* New ``best_fit_rotation`` and ``qinterp`` top-level methods.

Improvements
~~~~~~~~~~~~

* Added ``mode`` and ``outlier_thresh`` arguments to
  ``estimate_angular_velocity``.


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
