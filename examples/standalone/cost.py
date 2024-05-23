import numpy as np

import pybop


class StandaloneCost(pybop.BaseCost):
    """
    A standalone cost function example that inherits from pybop.BaseCost.

    This class represents a simple cost function without a problem object, used for demonstration purposes.
    It is a quadratic function of one variable with a constant term, defined by
    the formula: cost(x) = x^2 + 42.

    Parameters
    ----------
    problem : object, optional
        A dummy problem instance used to initialize the superclass. This is not
        used in the current class but is accepted for compatibility with the
        BaseCost interface.
    x0 : array-like
        The initial guess for the optimization problem, set to [4.2].
    _n_parameters : int
        The number of parameters in the model, which is 1 in this case.
    bounds : dict
        A dictionary containing the lower and upper bounds for the parameter,
        set to [-1] and [10], respectively.

    Methods
    -------
    __call__(x, grad=None)
        Calculate the cost for a given parameter value.
    """

    def __init__(self, problem=None):
        """
        Initialize the StandaloneCost class with optional problem instance.

        The problem parameter is not utilized in this subclass. The initial guess,
        number of parameters, and bounds are predefined for the standalone cost function.
        """
        super().__init__(problem)

        self.x0 = np.array([4.2])
        self._n_parameters = len(self.x0)

        self.bounds = dict(
            lower=[-1],
            upper=[10],
        )

    def _evaluate(self, x, grad=None):
        """
        Calculate the cost for a given parameter value.

        The cost function is defined as cost(x) = x^2 + 42, where x is the
        parameter value.

        Parameters
        ----------
        x : array-like
            A one-element array containing the parameter value for which to
            evaluate the cost.
        grad : array-like, optional
            Unused parameter, present for compatibility with gradient-based
            optimizers.

        Returns
        -------
        float
            The calculated cost value for the given parameter.
        """

        return x[0] ** 2 + 42
