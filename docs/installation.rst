.. highlight:: shell

============
Installation
============


Latest version
--------------

rigid-body-motion can be installed via ``pip``:

.. code-block:: console

    $ pip install rigid-body-motion

or via ``conda``:

.. code-block:: console

    $ conda install -c phausamann -c conda-forge rigid-body-motion


Optional dependencies
---------------------

Optional dependencies can be installed via ``pip`` or ``conda`` (just replace
``pip install`` with ``conda install``).

Transformations can be sped up significantly by installing the numba library:

.. code-block:: console

    $ pip install numba

rigid-body-motion supports xarray data types:

.. code-block:: console

    $ pip install xarray

Loading the example datasets requires the netCDF4 library:

.. code-block:: console

    $ pip install netcdf4

Plotting functions require matplotlib:

.. code-block:: console

    $ pip install matplotlib


Example environment
-------------------

We recommend using rigid_body_motion in a dedicated conda environment.
For the examples in the user guide, we provide an
:download:`environment file<_static/envs/example-env.yml>`
that you can download and set up with:

.. code-block:: console

    $ conda env create -f example-env.yml

If you download the examples as Jupyter notebooks, it is sufficient to have
the Jupyter notebook server installed in your base environment alongside
``nb_conda``:

.. code-block:: console

    $ conda install -n base nb_conda

Now you can open any of the example notebooks, go to *Kernel > Change kernel*
and select *Python [conda env:rbm-examples]*.

Pre-release version
-------------------

The latest pre-release can be installed from the GitHub master branch:

.. code-block:: console

    $ pip install git+https://github.com/phausamann/rigid-body-motion.git
