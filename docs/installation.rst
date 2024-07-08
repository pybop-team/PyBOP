.. _installation:

Installation
*****************************

PyBOP is a versatile Python package designed for optimisation and parameterisation of battery models. Follow the instructions below to install PyBOP and set up your environment to begin utilising its capabilities.

Installing PyBOP with pip
-------------------------

The simplest method to install PyBOP is using pip. Run the following command in your terminal:

.. code-block:: console

    pip install pybop

This command will download and install the latest stable version of PyBOP. If you want to install a specific version, you can specify the version number using the following command:

.. code-block:: console

    pip install pybop==23.11

Installing the Development Version
----------------------------------

If you're interested in the cutting-edge features and want to try out the latest enhancements, you can install the development version directly from the ``develop`` branch on GitHub:

.. code-block:: console

    pip install git+https://github.com/pybop-team/PyBOP.git@develop

Please note that the development version may be less stable than the official releases.

Local Installation from Source
------------------------------

For those who prefer to install PyBOP from a local clone of the repository or wish to modify the source code, you can use pip to install the package in "editable" mode. Replace "path/to/pybop" with the actual path to your local PyBOP directory:

.. code-block:: console

    pip install -e "path/to/pybop"

In editable mode, changes you make to the source code will immediately affect the PyBOP installation without the need for reinstallation.

Verifying Installation
----------------------

To verify that PyBOP has been installed successfully, try running one of the provided example scripts included in the documentation or repository. If the example executes without any errors, PyBOP is ready to use. Alternatively, you can run the following command in your terminal to check the version of PyBOP that is installed:

.. code-block:: console

    python -c "import pybop; print(pybop.__version__)"


For Developers
--------------

If you are installing PyBOP for development purposes, such as contributing to the project, please ensure that you follow the guidelines outlined in the `Contributing Guide <Contributing.html>`_. It includes additional steps that might be necessary for setting up a development environment, including the installation of dependencies and setup of pre-commit hooks.

Further Assistance
------------------

If you encounter any issues during the installation process or have any questions regarding the use of PyBOP, feel free to reach out to the community via the `PyBOP GitHub Discussions <https://github.com/pybop-team/PyBOP/discussions>`_.

Next Steps
----------

After installing PyBOP, you might want to:

* Explore the `Quick Start Guide <quick_start.html>`_ to begin using PyBOP.
* Check out the `API Reference <api/index.html>`_ for detailed information on PyBOP's programming interface.
