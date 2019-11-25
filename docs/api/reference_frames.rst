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

    ReferenceFrame.get_transformation
    ReferenceFrame.transform_points
    ReferenceFrame.transform_quaternions
    ReferenceFrame.transform_vectors

Registry
~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: _generated

    ReferenceFrame.register
    ReferenceFrame.deregister
