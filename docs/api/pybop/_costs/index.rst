:py:mod:`pybop._costs`
======================

.. py:module:: pybop._costs


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pybop._costs.BaseCost
   pybop._costs.RootMeanSquaredError
   pybop._costs.SumSquaredError




.. py:class:: BaseCost(problem)


   Base class for defining cost functions.

   This class is intended to be subclassed to create specific cost functions
   for evaluating model predictions against a set of data. The cost function
   quantifies the goodness-of-fit between the model predictions and the
   observed data, with a lower cost value indicating a better fit.

   :param problem: A problem instance containing the data and functions necessary for
                   evaluating the cost function.
   :type problem: object
   :param _target: An array containing the target data to fit.
   :type _target: array-like
   :param x0: The initial guess for the model parameters.
   :type x0: array-like
   :param bounds: The bounds for the model parameters.
   :type bounds: tuple
   :param n_parameters: The number of parameters in the model.
   :type n_parameters: int

   .. py:method:: __call__(x, grad=None)
      :abstractmethod:

      Calculate the cost function value for a given set of parameters.

      This method must be implemented by subclasses.

      :param x: The parameters for which to evaluate the cost.
      :type x: array-like
      :param grad: An array to store the gradient of the cost function with respect
                   to the parameters.
      :type grad: array-like, optional

      :returns: The calculated cost function value.
      :rtype: float

      :raises NotImplementedError: If the method has not been implemented by the subclass.



.. py:class:: RootMeanSquaredError(problem)


   Bases: :py:obj:`BaseCost`

   Root mean square error cost function.

   Computes the root mean square error between model predictions and the target
   data, providing a measure of the differences between predicted values and
   observed values.

   Inherits all parameters and attributes from ``BaseCost``.


   .. py:method:: __call__(x, grad=None)

      Calculate the root mean square error for a given set of parameters.

      :param x: The parameters for which to evaluate the cost.
      :type x: array-like
      :param grad: An array to store the gradient of the cost function with respect
                   to the parameters.
      :type grad: array-like, optional

      :returns: The root mean square error.
      :rtype: float

      :raises ValueError: If an error occurs during the calculation of the cost.



.. py:class:: SumSquaredError(problem)


   Bases: :py:obj:`BaseCost`

   Sum of squared errors cost function.

   Computes the sum of the squares of the differences between model predictions
   and target data, which serves as a measure of the total error between the
   predicted and observed values.

   Inherits all parameters and attributes from ``BaseCost``.

   Additional Attributes
   ---------------------
   _de : float
       The gradient of the cost function to use if an error occurs during
       evaluation. Defaults to 1.0.


   .. py:method:: __call__(x, grad=None)

      Calculate the sum of squared errors for a given set of parameters.

      :param x: The parameters for which to evaluate the cost.
      :type x: array-like
      :param grad: An array to store the gradient of the cost function with respect
                   to the parameters.
      :type grad: array-like, optional

      :returns: The sum of squared errors.
      :rtype: float

      :raises ValueError: If an error occurs during the calculation of the cost.


   .. py:method:: evaluateS1(x)

      Compute the cost and its gradient with respect to the parameters.

      :param x: The parameters for which to compute the cost and gradient.
      :type x: array-like

      :returns: A tuple containing the cost and the gradient. The cost is a float,
                and the gradient is an array-like of the same length as `x`.
      :rtype: tuple

      :raises ValueError: If an error occurs during the calculation of the cost or gradient.


   .. py:method:: set_fail_gradient(de)

      Set the fail gradient to a specified value.

      The fail gradient is used if an error occurs during the calculation
      of the gradient. This method allows updating the default gradient value.

      :param de: The new fail gradient value to be used.
      :type de: float
