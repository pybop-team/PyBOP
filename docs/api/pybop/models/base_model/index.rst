:py:mod:`pybop.models.base_model`
=================================

.. py:module:: pybop.models.base_model


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pybop.models.base_model.BaseModel




.. py:class:: BaseModel(name='Base Model')


   A base class for constructing and simulating models using PyBaMM.

   This class serves as a foundation for building specific models in PyBaMM.
   It provides methods to set up the model, define parameters, and perform
   simulations. The class is designed to be subclassed for creating models
   with custom behavior.

   .. method:: build(dataset=None, parameters=None, check_model=True, init_soc=None)

      Construct the PyBaMM model if not already built.

   .. method:: set_init_soc(init_soc)

      Set the initial state of charge for the battery model.

   .. method:: set_params()

      Assign the parameters to the model.

   .. method:: simulate(inputs, t_eval)

      Execute the forward model simulation and return the result.

   .. method:: simulateS1(inputs, t_eval)

      Perform the forward model simulation with sensitivities.

   .. method:: predict(inputs=None, t_eval=None, parameter_set=None, experiment=None, init_soc=None)

      Solve the model using PyBaMM's simulation framework and return the solution.


   .. py:property:: built_model


   .. py:property:: geometry


   .. py:property:: mesh


   .. py:property:: model_with_set_params


   .. py:property:: parameter_set


   .. py:property:: solver


   .. py:property:: spatial_methods


   .. py:property:: submesh_types


   .. py:property:: var_pts


   .. py:method:: build(dataset=None, parameters=None, check_model=True, init_soc=None)

      Construct the PyBaMM model if not already built, and set parameters.

      This method initializes the model components, applies the given parameters,
      sets up the mesh and discretization if needed, and prepares the model
      for simulations.

      :param dataset: The dataset to be used in the model construction.
      :type dataset: pybamm.Dataset, optional
      :param parameters: A dictionary containing parameter values to apply to the model.
      :type parameters: dict, optional
      :param check_model: If True, the model will be checked for correctness after construction.
      :type check_model: bool, optional
      :param init_soc: The initial state of charge to be used in simulations.
      :type init_soc: float, optional


   .. py:method:: predict(inputs=None, t_eval=None, parameter_set=None, experiment=None, init_soc=None)

      Solve the model using PyBaMM's simulation framework and return the solution.

      This method sets up a PyBaMM simulation by configuring the model, parameters, experiment
      (if any), and initial state of charge (if provided). It then solves the simulation and
      returns the resulting solution object.

      :param inputs: Input parameters for the simulation. If the input is array-like, it is converted
                     to a dictionary using the model's fitting keys. Defaults to None, indicating
                     that the default parameters should be used.
      :type inputs: dict or array-like, optional
      :param t_eval: An array of time points at which to evaluate the solution. Defaults to None,
                     which means the time points need to be specified within experiment or elsewhere.
      :type t_eval: array-like, optional
      :param parameter_set: A PyBaMM ParameterValues object or a dictionary containing the parameter values
                            to use for the simulation. Defaults to the model's current ParameterValues if None.
      :type parameter_set: pybamm.ParameterValues, optional
      :param experiment: A PyBaMM Experiment object specifying the experimental conditions under which
                         the simulation should be run. Defaults to None, indicating no experiment.
      :type experiment: pybamm.Experiment, optional
      :param init_soc: The initial state of charge for the simulation, as a fraction (between 0 and 1).
                       Defaults to None.
      :type init_soc: float, optional

      :returns: The solution object returned after solving the simulation.
      :rtype: pybamm.Solution

      :raises ValueError: If the model has not been configured properly before calling this method or
          if PyBaMM models are not supported by the current simulation method.


   .. py:method:: set_init_soc(init_soc)

      Set the initial state of charge for the battery model.

      :param init_soc: The initial state of charge to be used in the model.
      :type init_soc: float


   .. py:method:: set_params()

      Assign the parameters to the model.

      This method processes the model with the given parameters, sets up
      the geometry, and updates the model instance.


   .. py:method:: simulate(inputs, t_eval)

      Execute the forward model simulation and return the result.

      :param inputs: The input parameters for the simulation. If array-like, it will be
                     converted to a dictionary using the model's fit keys.
      :type inputs: dict or array-like
      :param t_eval: An array of time points at which to evaluate the solution.
      :type t_eval: array-like

      :returns: The simulation result corresponding to the specified signal.
      :rtype: array-like

      :raises ValueError: If the model has not been built before simulation.


   .. py:method:: simulateS1(inputs, t_eval)

      Perform the forward model simulation with sensitivities.

      :param inputs: The input parameters for the simulation. If array-like, it will be
                     converted to a dictionary using the model's fit keys.
      :type inputs: dict or array-like
      :param t_eval: An array of time points at which to evaluate the solution and its
                     sensitivities.
      :type t_eval: array-like

      :returns: A tuple containing the simulation result and the sensitivities.
      :rtype: tuple

      :raises ValueError: If the model has not been built before simulation.
