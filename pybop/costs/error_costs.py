import numpy as np
import pybop


class BaseCost:
    """
    Base class for defining cost functions.
    This class computes a corresponding goodness-of-fit for a corresponding model prediction and dataset.
    Lower cost values indicate a better fit.
    """

    def __call__(self, x):
        raise NotImplementedError

    def compute(self, x):
        """
        Calls the forward models and computes the cost.
        """
        raise NotImplementedError

    def n_parameters(self):
        """
        Returns the size of the parameter space.
        """
        raise NotImplementedError


class ProblemCost(BaseCost):
    """
    Extends the base cost function class for a single output problem.
    """

    def __init__(self, problem):
        super(ProblemCost, self).__init__()
        self._problem = problem
        self._target = problem._target

    def n_parameters(self):
        """
        Returns the dimension of the parameter space.
        """
        return self._problem.n_parameters()


class RootMeanSquaredError(ProblemCost):
    """
    Defines the root mean square error cost function.
    """

    def __init__(self, problem):
        super(RootMeanSquaredError, self).__init__(problem)

        if not isinstance(problem, pybop.Problem):
            raise ValueError("This cost function only supports pybop problems")

    def compute(self, x):
        # Compute the cost
        try:
            return np.sqrt(np.mean((self._problem.evaluate(x) - self._target) ** 2))

        except Exception as e:
            raise ValueError(f"Error in RMSE calculation: {e}")
