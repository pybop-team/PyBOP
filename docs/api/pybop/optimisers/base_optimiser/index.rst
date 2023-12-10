:py:mod:`pybop.optimisers.base_optimiser`
=========================================

.. py:module:: pybop.optimisers.base_optimiser


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pybop.optimisers.base_optimiser.BaseOptimiser




.. py:class:: BaseOptimiser


   A base class for defining optimisation methods.

   This class serves as a template for creating optimisers. It provides a basic structure for
   an optimisation algorithm, including the initial setup and a method stub for performing
   the optimisation process. Child classes should override the optimise and _runoptimise
   methods with specific algorithms.

   .. method:: optimise(cost_function, x0=None, bounds=None, maxiter=None)

      Initiates the optimisation process. This is a stub and should be implemented in child classes.

   .. method:: _runoptimise(cost_function, x0=None, bounds=None)

      Contains the logic for the optimisation algorithm. This is a stub and should be implemented in child classes.

   .. method:: name()

      Returns the name of the optimiser.


   .. py:method:: _runoptimise(cost_function, x0=None, bounds=None)

      Contains the logic for the optimisation algorithm.

      This method should be implemented by child classes to perform the actual optimisation.

      :param cost_function: The cost function to be minimised by the optimiser.
      :type cost_function: callable
      :param x0: Initial guess for the parameters. Default is None.
      :type x0: ndarray, optional
      :param bounds: Bounds on the parameters. Default is None.
      :type bounds: sequence or Bounds, optional

      :returns: * *This method is expected to return the result of the optimisation, the format of which*
                * *will be determined by the child class implementation.*


   .. py:method:: name()

      Returns the name of the optimiser.

      :returns: The name of the optimiser, which is "BaseOptimiser" for this base class.
      :rtype: str


   .. py:method:: optimise(cost_function, x0=None, bounds=None, maxiter=None)

      Initiates the optimisation process.

      This method should be overridden by child classes with the specific optimisation algorithm.

      :param cost_function: The cost function to be minimised by the optimiser.
      :type cost_function: callable
      :param x0: Initial guess for the parameters. Default is None.
      :type x0: ndarray, optional
      :param bounds: Bounds on the parameters. Default is None.
      :type bounds: sequence or Bounds, optional
      :param maxiter: Maximum number of iterations to perform. Default is None.
      :type maxiter: int, optional

      :rtype: The result of the optimisation process. The specific type of this result will depend on the child implementation.
