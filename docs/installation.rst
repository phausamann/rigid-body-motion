.. highlight:: shell

============
Installation
============


Latest version
--------------

rigid-body-motion can be installed via ``pip``:

.. code-block:: console

    $ pip install rigid-body-motion

If using the ``conda`` package manager, install these dependencies first:

.. code-block:: console

    $ conda install -c phausamann -c conda-forge rigid-body-motion

Afterwards, install the package via ``pip`` as detailed above.


Optional dependencies
---------------------

Transformations can be sped up significantly by installing the numba library:

.. code-block:: console

    $ pip install numba

or

.. code-block:: console

    $ conda install numba


rigid-body-motion supports xarray data types:

.. code-block:: console

    $ pip install xarray

or

.. code-block:: console

    $ conda install xarray


Pre-release version
-------------------

The latest pre-release can be installed from the GitHub master branch:

.. code-block:: console

    $ pip install git+https://github.com/phausamann/rigid-body-motion.git
