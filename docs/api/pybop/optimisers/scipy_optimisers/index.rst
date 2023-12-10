:py:mod:`pybop.optimisers.scipy_optimisers`
===========================================

.. py:module:: pybop.optimisers.scipy_optimisers


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pybop.optimisers.scipy_optimisers.SciPyDifferentialEvolution
   pybop.optimisers.scipy_optimisers.SciPyMinimize




.. py:class:: SciPyDifferentialEvolution(bounds=None, strategy='best1bin', maxiter=1000, popsize=15)


   Bases: :py:obj:`pybop.optimisers.base_optimiser.BaseOptimiser`

   Adapts SciPy's differential_evolution function for global optimization.

   This class provides a global optimization strategy based on differential evolution, useful for problems involving continuous parameters and potentially multiple local minima.

   :param bounds: Bounds for variables. Must be provided as it is essential for differential evolution.
   :type bounds: sequence or ``Bounds``
   :param strategy: The differential evolution strategy to use. Defaults to 'best1bin'.
   :type strategy: str, optional
   :param maxiter: Maximum number of iterations to perform. Defaults to 1000.
   :type maxiter: int, optional
   :param popsize: The number of individuals in the population. Defaults to 15.
   :type popsize: int, optional

   .. py:method:: _runoptimise(cost_function, x0=None, bounds=None)

      Executes the optimization process using SciPy's differential_evolution function.

      :param cost_function: The objective function to minimize.
      :type cost_function: callable
      :param x0: Ignored parameter, provided for API consistency.
      :type x0: array_like, optional
      :param bounds: Bounds for the variables, required for differential evolution.
      :type bounds: sequence or ``Bounds``

      :returns: A tuple (x, final_cost) containing the optimized parameters and the value of ``cost_function`` at the optimum.
      :rtype: tuple


   .. py:method:: name()

      Provides the name of the optimization strategy.

      :returns: The name 'SciPyDifferentialEvolution'.
      :rtype: str


   .. py:method:: needs_sensitivities()

      Determines if the optimization algorithm requires gradient information.

      :returns: False, indicating that gradient information is not required for differential evolution.
      :rtype: bool



.. py:class:: SciPyMinimize(method=None, bounds=None, maxiter=None)


   Bases: :py:obj:`pybop.optimisers.base_optimiser.BaseOptimiser`

   Adapts SciPy's minimize function for use as an optimization strategy.

   This class provides an interface to various scalar minimization algorithms implemented in SciPy, allowing fine-tuning of the optimization process through method selection and option configuration.

   :param method: The type of solver to use. If not specified, defaults to 'COBYLA'.
   :type method: str, optional
   :param bounds: Bounds for variables as supported by the selected method.
   :type bounds: sequence or ``Bounds``, optional
   :param maxiter: Maximum number of iterations to perform.
   :type maxiter: int, optional

   .. py:method:: _runoptimise(cost_function, x0, bounds)

      Executes the optimization process using SciPy's minimize function.

      :param cost_function: The objective function to minimize.
      :type cost_function: callable
      :param x0: Initial guess for the parameters.
      :type x0: array_like
      :param bounds: Bounds for the variables.
      :type bounds: sequence or `Bounds`

      :returns: A tuple (x, final_cost) containing the optimized parameters and the value of `cost_function` at the optimum.
      :rtype: tuple


   .. py:method:: name()

      Provides the name of the optimization strategy.

      :returns: The name 'SciPyMinimize'.
      :rtype: str


   .. py:method:: needs_sensitivities()

      Determines if the optimization algorithm requires gradient information.

      :returns: False, indicating that gradient information is not required.
      :rtype: bool
