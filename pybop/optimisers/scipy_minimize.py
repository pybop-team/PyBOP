import pybop
from scipy.optimize import minimize
from .base_optimiser import BaseOptimiser


class SciPyMinimize(BaseOptimiser):
    """
    Wrapper class for the SciPy optimisation class. Extends the BaseOptimiser class.
    """

    def __init__(self, cost_function, x0, bounds=None, options=None):
        super().__init__()
        self.cost_function = cost_function
        self.method = options.optmethod
        self.x0 = x0 or cost_function.x0
        self.bounds = bounds
        self.options = options
        self.name = "Scipy Optimiser"

    def _runoptimise(self):
        """
        Run the SciPy optimisation method.

        Inputs
        ----------
        cost_function: function for optimising
        method: optimisation algorithm
        x0: initialisation array
        bounds: bounds array
        """

        if self.method is not None:
            method=self.method
        else:
            opt = minimize(self.cost_function, self.x0, method="BFGS")

        # Reformat bounds
        bounds = (
            (lower, upper) for lower, upper in zip(bounds["lower"], bounds["upper"])
        )

        # Run the optimser
        if self.bounds is not None:
            output = minimize(
                self.cost_function, self.x0, method=method, bounds=self.bounds, tol=self.xtol
            )
        else:
            output = minimize(self.cost_function, self.x0, method=method, tol=self.xtol)

        # Get performance statistics
        x = output.x
        final_cost = output.fun
        num_evals = output.nfev

        return x, output, final_cost, num_evals
