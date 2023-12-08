:py:mod:`pybop.optimisers.nlopt_optimize`
=========================================

.. py:module:: pybop.optimisers.nlopt_optimize


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pybop.optimisers.nlopt_optimize.NLoptOptimize




.. py:class:: NLoptOptimize(n_param, xtol=None, method=None)


   Bases: :py:obj:`pybop.optimisers.base_optimiser.BaseOptimiser`

   Wrapper class for the NLOpt optimiser class. Extends the BaseOptimiser class.

   .. py:method:: _runoptimise(cost_function, x0, bounds)

      Run the NLOpt optimisation method.

      Inputs
      ----------
      cost_function: function for optimising
      method: optimisation algorithm
      x0: initialisation array
      bounds: bounds array


   .. py:method:: name()

      Returns the name of the optimiser.


   .. py:method:: needs_sensitivities()

      Returns True if the optimiser needs sensitivities.
