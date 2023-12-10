:py:mod:`pybop.optimisers.pints_optimisers`
===========================================

.. py:module:: pybop.optimisers.pints_optimisers


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pybop.optimisers.pints_optimisers.Adam
   pybop.optimisers.pints_optimisers.CMAES
   pybop.optimisers.pints_optimisers.GradientDescent
   pybop.optimisers.pints_optimisers.IRPropMin
   pybop.optimisers.pints_optimisers.PSO
   pybop.optimisers.pints_optimisers.SNES
   pybop.optimisers.pints_optimisers.XNES




.. py:class:: Adam(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.Adam`

   Implements the Adam optimization algorithm.

   This class extends the Adam optimizer from the PINTS library, which combines
   ideas from RMSProp and Stochastic Gradient Descent with momentum. Note that
   this optimizer does not support boundary constraints.

   :param x0: Initial position from which optimization will start.
   :type x0: array_like
   :param sigma0: Initial step size (default is 0.1).
   :type sigma0: float, optional
   :param bounds: Ignored by this optimizer, provided for API consistency.
   :type bounds: sequence or ``Bounds``, optional

   .. seealso::

      :obj:`pints.Adam`
          The PINTS implementation this class is based on.


.. py:class:: CMAES(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.CMAES`

   Adapter for the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimizer in PINTS.

   CMA-ES is an evolutionary algorithm for difficult non-linear non-convex optimization problems.
   It adapts the covariance matrix of a multivariate normal distribution to capture the shape of the cost landscape.

   :param x0: The initial parameter vector to optimize.
   :type x0: array_like
   :param sigma0: Initial standard deviation of the sampling distribution, defaults to 0.1.
   :type sigma0: float, optional
   :param bounds: A dictionary with 'lower' and 'upper' keys containing arrays for lower and upper bounds on the parameters.
                  If ``None``, no bounds are enforced.
   :type bounds: dict, optional

   .. seealso::

      :obj:`pints.CMAES`
          PINTS implementation of CMA-ES algorithm.


.. py:class:: GradientDescent(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.GradientDescent`

   Implements a simple gradient descent optimization algorithm.

   This class extends the gradient descent optimizer from the PINTS library, designed
   to minimize a scalar function of one or more variables. Note that this optimizer
   does not support boundary constraints.

   :param x0: Initial position from which optimization will start.
   :type x0: array_like
   :param sigma0: Initial step size (default is 0.1).
   :type sigma0: float, optional
   :param bounds: Ignored by this optimizer, provided for API consistency.
   :type bounds: sequence or ``Bounds``, optional

   .. seealso::

      :obj:`pints.GradientDescent`
          The PINTS implementation this class is based on.


.. py:class:: IRPropMin(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.IRPropMin`

   Implements the iRpropMin optimization algorithm.

   This class inherits from the PINTS IRPropMin class, which is an optimizer that
   uses resilient backpropagation with weight-backtracking. It is designed to handle
   problems with large plateaus, noisy gradients, and local minima.

   :param x0: Initial position from which optimization will start.
   :type x0: array_like
   :param sigma0: Initial step size (default is 0.1).
   :type sigma0: float, optional
   :param bounds: Lower and upper bounds for each optimization parameter.
   :type bounds: dict, optional

   .. seealso::

      :obj:`pints.IRPropMin`
          The PINTS implementation this class is based on.


.. py:class:: PSO(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.PSO`

   Implements a particle swarm optimization (PSO) algorithm.

   This class extends the PSO optimizer from the PINTS library. PSO is a
   metaheuristic optimization method inspired by the social behavior of birds
   flocking or fish schooling, suitable for global optimization problems.

   :param x0: Initial positions of particles, which the optimization will use.
   :type x0: array_like
   :param sigma0: Spread of the initial particle positions (default is 0.1).
   :type sigma0: float, optional
   :param bounds: Lower and upper bounds for each optimization parameter.
   :type bounds: dict, optional

   .. seealso::

      :obj:`pints.PSO`
          The PINTS implementation this class is based on.


.. py:class:: SNES(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.SNES`

   Implements the stochastic natural evolution strategy (SNES) optimization algorithm.

   Inheriting from the PINTS SNES class, this optimizer is an evolutionary algorithm
   that evolves a probability distribution on the parameter space, guiding the search
   for the optimum based on the natural gradient of expected fitness.

   :param x0: Initial position from which optimization will start.
   :type x0: array_like
   :param sigma0: Initial step size (default is 0.1).
   :type sigma0: float, optional
   :param bounds: Lower and upper bounds for each optimization parameter.
   :type bounds: dict, optional

   .. seealso::

      :obj:`pints.SNES`
          The PINTS implementation this class is based on.


.. py:class:: XNES(x0, sigma0=0.1, bounds=None)


   Bases: :py:obj:`pints.XNES`

   Implements the Exponential Natural Evolution Strategy (XNES) optimizer from PINTS.

   XNES is an evolutionary algorithm that samples from a multivariate normal distribution, which is updated iteratively to fit the distribution of successful solutions.

   :param x0: The initial parameter vector to optimize.
   :type x0: array_like
   :param sigma0: Initial standard deviation of the sampling distribution, defaults to 0.1.
   :type sigma0: float, optional
   :param bounds: A dictionary with 'lower' and 'upper' keys containing arrays for lower and upper bounds on the parameters. If ``None``, no bounds are enforced.
   :type bounds: dict, optional

   .. seealso::

      :obj:`pints.XNES`
          PINTS implementation of XNES algorithm.
