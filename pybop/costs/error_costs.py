import numpy as np


class BaseCost:
    """
    Base class for defining cost functions.
    This class computes a corresponding goodness-of-fit for a corresponding model prediction and dataset.
    Lower cost values indicate a better fit.
    """

    def __init__(self, problem):
        self.problem = problem
        self._target = problem._target
        self.x0 = problem.x0
        self.bounds = problem.bounds
        self.n_parameters = problem.n_parameters

    def __call__(self, x, grad=None):
        """
        Returns the cost function value and computes the cost.
        """
        raise NotImplementedError


class RootMeanSquaredError(BaseCost):
    """
    Defines the root mean square error cost function.
    """

    def __init__(self, problem):
        super(RootMeanSquaredError, self).__init__(problem)

    def __call__(self, x, grad=None):
        """
        Computes the cost.
        """
        try:
            return np.sqrt(np.mean((self.problem.evaluate(x) - self._target) ** 2))

        except Exception as e:
            raise ValueError(f"Error in cost calculation: {e}")


class SumSquaredError(BaseCost):
    """
    Defines the sum squared error cost function.
    """

    def __init__(self, problem):
        super(SumSquaredError, self).__init__(problem)

    def __call__(self, x, grad=None):
        """
        Computes the cost.
        """
        try:
            return np.sum(
                (np.sum(((self.problem.evaluate(x) - self._target) ** 2), axis=0)),
                axis=0,
            )
        except Exception as e:
            raise ValueError(f"Error in cost calculation: {e}")

    def evaluateS1(self, x):
        """
        Compute the cost and corresponding
        gradients with respect to the parameters.
        """
        try:
            y, dy = self.problem.evaluateS1(x)
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
