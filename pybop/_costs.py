import numpy as np


class BaseCost:
    """
    Base class for defining cost functions.

    This class is intended to be subclassed to create specific cost functions
    for evaluating model predictions against a set of data. The cost function
    quantifies the goodness-of-fit between the model predictions and the
    observed data, with a lower cost value indicating a better fit.

    Parameters
    ----------
    problem : object
        A problem instance containing the data and functions necessary for
        evaluating the cost function.
    _target : array-like
        An array containing the target data to fit.
    x0 : array-like
        The initial guess for the model parameters.
    bounds : tuple
        The bounds for the model parameters.
    n_parameters : int
        The number of parameters in the model.
    """

    def __init__(self, problem):
        self.problem = problem
        if problem is not None:
            self._target = problem._target
            self.x0 = problem.x0
            self.bounds = problem.bounds
            self.n_parameters = problem.n_parameters

    def __call__(self, x, grad=None):
        """
        Calculate the cost function value for a given set of parameters.

        This method must be implemented by subclasses.

        Parameters
        ----------
        x : array-like
            The parameters for which to evaluate the cost.
        grad : array-like, optional
            An array to store the gradient of the cost function with respect
            to the parameters.

        Returns
        -------
        float
            The calculated cost function value.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the subclass.
        """

        raise NotImplementedError


class RootMeanSquaredError(BaseCost):
    """
    Root mean square error cost function.

    Computes the root mean square error between model predictions and the target
    data, providing a measure of the differences between predicted values and
    observed values.

    Inherits all parameters and attributes from ``BaseCost``.

    """

    def __init__(self, problem):
        super(RootMeanSquaredError, self).__init__(problem)

    def __call__(self, x, grad=None):
        """
        Calculate the root mean square error for a given set of parameters.

        Parameters
        ----------
        x : array-like
            The parameters for which to evaluate the cost.
        grad : array-like, optional
            An array to store the gradient of the cost function with respect
            to the parameters.

        Returns
        -------
        float
            The root mean square error.

        Raises
        ------
        ValueError
            If an error occurs during the calculation of the cost.
        """

        try:
            prediction = self.problem.evaluate(x)

            if len(prediction) < len(self._target):
                return np.float64(np.inf)  # simulation stopped early
            else:
                return np.sqrt(np.mean((prediction - self._target) ** 2))

        except Exception as e:
            raise ValueError(f"Error in cost calculation: {e}")


class SumSquaredError(BaseCost):
    """
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

    """

    def __init__(self, problem):
        super(SumSquaredError, self).__init__(problem)

        # Default fail gradient
        self._de = 1.0

    def __call__(self, x, grad=None):
        """
        Calculate the sum of squared errors for a given set of parameters.

        Parameters
        ----------
        x : array-like
            The parameters for which to evaluate the cost.
        grad : array-like, optional
            An array to store the gradient of the cost function with respect
            to the parameters.

        Returns
        -------
        float
            The sum of squared errors.

        Raises
        ------
        ValueError
            If an error occurs during the calculation of the cost.
        """
        try:
            prediction = self.problem.evaluate(x)

            if len(prediction) < len(self._target):
                return np.float64(np.inf)  # simulation stopped early
            else:
                return np.sum(
                    (np.sum(((prediction - self._target) ** 2), axis=0)),
                    axis=0,
                )

        except Exception as e:
            raise ValueError(f"Error in cost calculation: {e}")

    def evaluateS1(self, x):
        """
        Compute the cost and its gradient with respect to the parameters.

        Parameters
        ----------
        x : array-like
            The parameters for which to compute the cost and gradient.

        Returns
        -------
        tuple
            A tuple containing the cost and the gradient. The cost is a float,
            and the gradient is an array-like of the same length as `x`.

        Raises
        ------
        ValueError
            If an error occurs during the calculation of the cost or gradient.
        """
        try:
            y, dy = self.problem.evaluateS1(x)
            if len(y) < len(self._target):
                e = np.float64(np.inf)
                de = self._de * np.ones(self.problem.n_parameters)
            else:
                dy = dy.reshape(
                    (
                        self.problem.n_time_data,
                        self.problem.n_outputs,
                        self.problem.n_parameters,
                    )
                )
                r = y - self._target
                e = np.sum(np.sum(r**2, axis=0), axis=0)
                de = 2 * np.sum(np.sum((r.T * dy.T), axis=2), axis=1)

            return e, de

        except Exception as e:
            raise ValueError(f"Error in cost calculation: {e}")

    def set_fail_gradient(self, de):
        """
        Set the fail gradient to a specified value.

        The fail gradient is used if an error occurs during the calculation
        of the gradient. This method allows updating the default gradient value.

        Parameters
        ----------
        de : float
            The new fail gradient value to be used.
        """
        de = float(de)
        self._de = de
