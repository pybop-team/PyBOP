:py:mod:`pybop.optimisers.nlopt_optimize`
=========================================

.. py:module:: pybop.optimisers.nlopt_optimize


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pybop.optimisers.nlopt_optimize.NLoptOptimize




.. py:class:: NLoptOptimize(n_param, xtol=None, method=None, maxiter=None)


   Bases: :py:obj:`pybop.optimisers.base_optimiser.BaseOptimiser`

   Extends BaseOptimiser to utilize the NLopt library for nonlinear optimization.

   This class serves as an interface to the NLopt optimization algorithms. It allows the user to
   define an optimization problem with bounds, initial guesses, and to select an optimization method
   provided by NLopt.

   :param n_param: Number of parameters to optimize.
   :type n_param: int
   :param xtol: The relative tolerance for optimization (stopping criteria). If not provided, a default of 1e-5 is used.
   :type xtol: float, optional
   :param method: The NLopt algorithm to use for optimization. If not provided, LN_BOBYQA is used by default.
   :type method: nlopt.algorithm, optional
   :param maxiter: The maximum number of iterations to perform during optimization. If not provided, NLopt's default is used.
   :type maxiter: int, optional

   .. method:: _runoptimise(cost_function, x0, bounds)

      Performs the optimization using the NLopt library.

   .. method:: needs_sensitivities()

      Indicates whether the optimizer requires gradient information.

   .. method:: name()

      Returns the name of the optimizer.


   .. py:method:: _runoptimise(cost_function, x0, bounds)

      Runs the optimization process using the NLopt library.

      :param cost_function: The objective function to minimize. It should take an array of parameter values and return the scalar cost.
      :type cost_function: callable
      :param x0: The initial guess for the parameters.
      :type x0: array_like
      :param bounds: A dictionary containing the 'lower' and 'upper' bounds arrays for the parameters.
      :type bounds: dict

      :returns: A tuple containing the optimized parameter values and the final cost.
      :rtype: tuple


   .. py:method:: name()

      Returns the name of this optimizer instance.

      :returns: The name 'NLoptOptimize' representing this NLopt optimization class.
      :rtype: str


   .. py:method:: needs_sensitivities()

      Indicates if the optimizer requires gradient information for the cost function.

      :returns: False, as the default NLopt algorithms do not require gradient information.
      :rtype: bool
