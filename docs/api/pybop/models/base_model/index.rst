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


   Base class for pybop models.

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

      Build the PyBOP model (if not built already).
      For PyBaMM forward models, this method follows a
      similar process to pybamm.Simulation.build().


   .. py:method:: predict(inputs=None, t_eval=None, parameter_set=None, experiment=None, init_soc=None)

      Create a PyBaMM simulation object, solve it, and return a solution object.


   .. py:method:: set_init_soc(init_soc)

      Set the initial state of charge.


   .. py:method:: set_params()

      Set the parameters in the model.


   .. py:method:: simulate(inputs, t_eval)

      Run the forward model and return the result in Numpy array format
      aligning with Pints' ForwardModel simulate method.


   .. py:method:: simulateS1(inputs, t_eval)

      Run the forward model and return the function evaulation and it's gradient
      aligning with Pints' ForwardModel simulateS1 method.
