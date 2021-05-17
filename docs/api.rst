.. _api_reference:

API Reference
=============

Top-level functions
-------------------

.. currentmodule:: rigid_body_motion

Module: :py:mod:`rigid_body_motion`

Transformations
~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: _generated

    transform_vectors
    transform_points
    transform_quaternions
    transform_coordinates
    transform_angular_velocity
    transform_linear_velocity

Reference Frames
~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: _generated

    ReferenceFrame
    register_frame
    deregister_frame
    clear_registry
    render_tree

Coordinate Systems
~~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: _generated

    cartesian_to_polar
    polar_to_cartesian
    cartesian_to_spherical
    spherical_to_cartesian

Lookups
~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: _generated

    lookup_transform
    lookup_pose
    lookup_twist
    lookup_linear_velocity
    lookup_angular_velocity

Estimators
~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: _generated

    estimate_linear_velocity
    estimate_angular_velocity
    shortest_arc_rotation
    best_fit_rotation
    best_fit_transform
    iterative_closest_point


Utility functions
~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: _generated

    qinv
    qmul
    qmean
    qinterp
    rotate_vectors


I/O functions
-------------

.. currentmodule:: rigid_body_motion.io

.. autosummary::
    :nosignatures:
    :toctree: _generated

    load_optitrack


Plotting
--------

.. currentmodule:: rigid_body_motion.plot

.. autosummary::
    :nosignatures:
    :toctree: _generated

    reference_frame
    points
    quaternions
    vectors


xarray Accessors
----------------

.. currentmodule:: xarray

.. autosummary::
    :toctree: _generated
    :template: autosummary/accessor_method.rst

    DataArray.rbm.qinterp
    DataArray.rbm.qinv



Class member details
--------------------

.. toctree::
    :maxdepth: 1

    api/reference_frames
