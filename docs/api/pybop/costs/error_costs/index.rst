:py:mod:`pybop.costs.error_costs`
=================================

.. py:module:: pybop.costs.error_costs


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pybop.costs.error_costs.BaseCost
   pybop.costs.error_costs.RootMeanSquaredError
   pybop.costs.error_costs.SumSquaredError




.. py:class:: BaseCost(problem)


   Base class for defining cost functions.
   This class computes a corresponding goodness-of-fit for a corresponding model prediction and dataset.
   Lower cost values indicate a better fit.

   .. py:method:: __call__(x, grad=None)
      :abstractmethod:

      Returns the cost function value and computes the cost.



.. py:class:: RootMeanSquaredError(problem)


   Bases: :py:obj:`BaseCost`

   Defines the root mean square error cost function.

   .. py:method:: __call__(x, grad=None)

      Computes the cost.



.. py:class:: SumSquaredError(problem)


   Bases: :py:obj:`BaseCost`

   Defines the sum squared error cost function.

   The initial fail gradient is set equal to one, but this can be
   changed at any time with :meth:`set_fail_gradient()`.

   .. py:method:: __call__(x, grad=None)

      Computes the cost.


   .. py:method:: evaluateS1(x)

      Compute the cost and corresponding
      gradients with respect to the parameters.


   .. py:method:: set_fail_gradient(de)

      Sets the fail gradient for this optimiser.
