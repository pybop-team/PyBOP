:py:mod:`pybop.optimisers.scipy_minimize`
=========================================

.. py:module:: pybop.optimisers.scipy_minimize


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pybop.optimisers.scipy_minimize.SciPyMinimize




.. py:class:: SciPyMinimize(method=None, bounds=None)


   Bases: :py:obj:`pybop.optimisers.base_optimiser.BaseOptimiser`

   Wrapper class for the SciPy optimisation class. Extends the BaseOptimiser class.

   .. py:method:: _runoptimise(cost_function, x0, bounds)

      Run the SciPy optimisation method.

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
