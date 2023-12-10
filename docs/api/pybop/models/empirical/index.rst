:py:mod:`pybop.models.empirical`
================================

.. py:module:: pybop.models.empirical


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   ecm/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   pybop.models.empirical.Thevenin




.. py:class:: Thevenin(name='Equivalent Circuit Thevenin Model', parameter_set=None, geometry=None, submesh_types=None, var_pts=None, spatial_methods=None, solver=None, options=None, **kwargs)


   Bases: :py:obj:`pybop.models.base_model.BaseModel`

   The Thevenin class represents an equivalent circuit model based on the Thevenin model in PyBaMM.

   This class encapsulates the PyBaMM equivalent circuit Thevenin model, providing an interface
   to define the parameters, geometry, submesh types, variable points, spatial methods, and solver
   to be used for simulations.

   :param name: A name for the model instance. Defaults to "Equivalent Circuit Thevenin Model".
   :type name: str, optional
   :param parameter_set: A dictionary of parameters to be used for the model. If None, the default parameters from PyBaMM are used.
   :type parameter_set: dict or None, optional
   :param geometry: The geometry definitions for the model. If None, the default geometry from PyBaMM is used.
   :type geometry: dict or None, optional
   :param submesh_types: The types of submeshes to use. If None, the default submesh types from PyBaMM are used.
   :type submesh_types: dict or None, optional
   :param var_pts: The number of points for each variable in the model to define the discretization. If None, the default is used.
   :type var_pts: dict or None, optional
   :param spatial_methods: The spatial methods to be used for discretization. If None, the default spatial methods from PyBaMM are used.
   :type spatial_methods: dict or None, optional
   :param solver: The solver to use for simulating the model. If None, the default solver from PyBaMM is used.
   :type solver: pybamm.Solver or None, optional
   :param options: A dictionary of options to pass to the PyBaMM Thevenin model.
   :type options: dict or None, optional
   :param \*\*kwargs: Additional arguments passed to the PyBaMM Thevenin model constructor.
