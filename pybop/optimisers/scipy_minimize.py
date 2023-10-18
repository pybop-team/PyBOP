import pybop
from scipy.optimize import minimize
from pybop.optimisers.base_optimiser import BaseOptimiser


class SciPyMinimize(BaseOptimiser):
    """
    Wrapper class for the SciPy optimisation class. Extends the BaseOptimiser class.
    """

    def __init__(self, x0, xtol=None, method=None, options=None):
        super().__init__()
        self.name = "Scipy Optimiser"

        if method is None:
            self.method = method
        else:
            self.method = "BFGS"

        if xtol is not None:
            self.xtol = xtol
        else:
            self.xtol = 1e-5

        self.options = options

    def _runoptimise(self, cost_function, x0, bounds):
        """
        Run the SciPy optimisation method.

        Inputs
        ----------
        cost_function: function for optimising
        method: optimisation algorithm
        x0: initialisation array
        bounds: bounds array
        """

        # Reformat bounds
        bounds = ((lower, upper) for lower, upper in zip(bounds["lower"], bounds["upper"]))

        # Run the optimser
        output = minimize(cost_function, x0, method=self.method, bounds=bounds, tol=self.xtol)

        # Get performance statistics
        x = output.x
        final_cost = output.fun
        num_evals = output.nfev

        return x, output, final_cost, num_evals
