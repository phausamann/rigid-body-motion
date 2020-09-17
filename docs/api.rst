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

    register_frame
    deregister_frame
    clear_registry
    ReferenceFrame

Coordinate Systems
~~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: _generated

    cartesian_to_polar
    polar_to_cartesian
    cartesian_to_spherical
    spherical_to_cartesian

Estimators
~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: _generated

    lookup_transform
    lookup_twist
    shortest_arc_rotation
    best_fit_transform
    iterative_closest_point


Utility functions
~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: _generated

    qinv
    qmean
    rotate_vectors

Class member details
--------------------

.. toctree::
    :maxdepth: 1

    api/reference_frames
