import numpy as np

from pybop import BaseProblem


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
    sigma0 : scalar or array
        Initial standard deviation around ``x0``. Either a scalar value (one
        standard deviation for all coordinates) or an array with one entry
        per dimension. Not all methods will use this information.
    _n_parameters : int
        The number of parameters in the model.
    n_outputs : int
        The number of outputs in the model.
    """

    def __init__(self, problem=None, sigma=None):
        self.problem = problem
        self.x0 = None
        self.bounds = None
        self.sigma0 = sigma
        if isinstance(self.problem, BaseProblem):
            self._target = problem._target
            self.parameters = problem.parameters
            self.x0 = problem.x0
            self.bounds = problem.bounds
            self.n_outputs = problem.n_outputs
            self.signal = problem.signal
            self._n_parameters = problem.n_parameters
            self.sigma0 = sigma or problem.sigma0 or np.zeros(self._n_parameters)

    @property
    def n_parameters(self):
        return self._n_parameters

    def __call__(self, x, grad=None):
        """
        Call the evaluate function for a given set of parameters.
        """
        return self.evaluate(x, grad)

    def evaluate(self, x, grad=None):
        """
        Call the evaluate function for a given set of parameters.

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
        ValueError
            If an error occurs during the calculation of the cost.
        """
        try:
            return self._evaluate(x, grad)

        except NotImplementedError as e:
            raise e

        except Exception as e:
            raise ValueError(f"Error in cost calculation: {e}")

    def _evaluate(self, x, grad=None):
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

    def evaluateS1(self, x):
        """
        Call _evaluateS1 for a given set of parameters.

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
            return self._evaluateS1(x)

        except NotImplementedError as e:
            raise e

        except Exception as e:
            raise ValueError(f"Error in cost calculation: {e}")

    def _evaluateS1(self, x):
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
        NotImplementedError
            If the method has not been implemented by the subclass.
        """
        raise NotImplementedError
