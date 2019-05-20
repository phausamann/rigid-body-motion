=================
Rigid Body Motion
=================


.. image:: https://img.shields.io/pypi/v/rigid-body-motion.svg
        :target: https://pypi.python.org/pypi/rigid-body-motion

.. image:: https://img.shields.io/travis/phausamann/rigid-body-motion.svg
        :target: https://travis-ci.com/phausamann/rigid-body-motion

.. image:: https://readthedocs.org/projects/rigid-body-motion/badge/?version=latest
        :target: https://rigid-body-motion.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Python utilities for estimating and transforming rigid body motion.


Documentation: https://rigid-body-motion.readthedocs.io.


Features
--------

* Transform numpy arrays describing rigid body motion between different coordinate systems and reference frames


Example
-------

    import numpy as np
    import rigid_body_motion as rbm

    arr = np.ones((10, 2))
    rbm.transform(arr).from_('cartesian').to_('polar')

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
