:py:mod:`pybop.models.lithium_ion.echem`
========================================

.. py:module:: pybop.models.lithium_ion.echem


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pybop.models.lithium_ion.echem.SPM
   pybop.models.lithium_ion.echem.SPMe




.. py:class:: SPM(name='Single Particle Model', parameter_set=None, geometry=None, submesh_types=None, var_pts=None, spatial_methods=None, solver=None, options=None)


   Bases: :py:obj:`pybop.models.base_model.BaseModel`

   Wraps the Single Particle Model (SPM) for simulating lithium-ion batteries, as implemented in PyBaMM.

   The SPM is a simplified physics-based model that represents a lithium-ion cell using a single
   spherical particle to simulate the behavior of the negative and positive electrodes.

   :param name: The name for the model instance, defaulting to "Single Particle Model".
   :type name: str, optional
   :param parameter_set: The parameters for the model. If None, default parameters provided by PyBaMM are used.
   :type parameter_set: pybamm.ParameterValues or dict, optional
   :param geometry: The geometry definitions for the model. If None, default geometry from PyBaMM is used.
   :type geometry: dict, optional
   :param submesh_types: The types of submeshes to use. If None, default submesh types from PyBaMM are used.
   :type submesh_types: dict, optional
   :param var_pts: The discretization points for each variable in the model. If None, default points from PyBaMM are used.
   :type var_pts: dict, optional
   :param spatial_methods: The spatial methods used for discretization. If None, default spatial methods from PyBaMM are used.
   :type spatial_methods: dict, optional
   :param solver: The solver to use for simulating the model. If None, the default solver from PyBaMM is used.
   :type solver: pybamm.Solver, optional
   :param options: A dictionary of options to customize the behavior of the PyBaMM model.
   :type options: dict, optional


.. py:class:: SPMe(name='Single Particle Model with Electrolyte', parameter_set=None, geometry=None, submesh_types=None, var_pts=None, spatial_methods=None, solver=None, options=None)


   Bases: :py:obj:`pybop.models.base_model.BaseModel`

   Represents the Single Particle Model with Electrolyte (SPMe) for lithium-ion batteries.

   The SPMe extends the basic Single Particle Model (SPM) by incorporating electrolyte dynamics,
   making it suitable for simulations where electrolyte effects are non-negligible. This class
   provides a framework to define the model parameters, geometry, mesh types, discretization
   points, spatial methods, and numerical solvers for simulation within the PyBaMM ecosystem.

   :param name: A name for the model instance, defaults to "Single Particle Model with Electrolyte".
   :type name: str, optional
   :param parameter_set: A dictionary or a ParameterValues object containing the parameters for the model. If None, the default PyBaMM parameters for SPMe are used.
   :type parameter_set: pybamm.ParameterValues or dict, optional
   :param geometry: A dictionary defining the model's geometry. If None, the default PyBaMM geometry for SPMe is used.
   :type geometry: dict, optional
   :param submesh_types: A dictionary defining the types of submeshes to use. If None, the default PyBaMM submesh types for SPMe are used.
   :type submesh_types: dict, optional
   :param var_pts: A dictionary specifying the number of points for each variable for discretization. If None, the default PyBaMM variable points for SPMe are used.
   :type var_pts: dict, optional
   :param spatial_methods: A dictionary specifying the spatial methods for discretization. If None, the default PyBaMM spatial methods for SPMe are used.
   :type spatial_methods: dict, optional
   :param solver: The solver to use for simulating the model. If None, the default PyBaMM solver for SPMe is used.
   :type solver: pybamm.Solver, optional
   :param options: A dictionary of options to customize the behavior of the PyBaMM model.
   :type options: dict, optional
