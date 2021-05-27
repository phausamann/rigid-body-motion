reference_frames
================

.. currentmodule:: rigid_body_motion

ReferenceFrame
--------------

Construction
~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: _generated

    ReferenceFrame.from_dataset
    ReferenceFrame.from_translation_dataarray
    ReferenceFrame.from_rotation_dataarray
    ReferenceFrame.from_rotation_matrix

Transforms
~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: _generated

    ReferenceFrame.transform_points
    ReferenceFrame.transform_quaternions
    ReferenceFrame.transform_vectors
    ReferenceFrame.transform_angular_velocity
    ReferenceFrame.transform_linear_velocity

Lookups
~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: _generated

    ReferenceFrame.lookup_transform
    ReferenceFrame.lookup_angular_velocity
    ReferenceFrame.lookup_linear_velocity
    ReferenceFrame.lookup_twist


Registry
~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: _generated

    ReferenceFrame.register
    ReferenceFrame.deregister
