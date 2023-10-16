import pybop
from scipy.optimize import minimize
from pybop.optimisers.base_optimiser import BaseOptimiser


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
        self.name = "Scipy Optimisation"

    def _runoptimise(self):
        """
        Run the SciPy optimisation method.

        Parameters
        ----------
        cost_function: function for optimising
        method: optimisation algorithm
        x0: initialisation array
        options: options dictionary
        bounds: bounds array
        """

        if self.method is not None and self.bounds is not None:
            opt = minimize(
                self.cost_function, self.x0, method=self.method, bounds=self.bounds
            )
        elif self.method is not None:
            opt = minimize(self.cost_function, self.x0, method=self.method)
        else:
            opt = minimize(self.cost_function, self.x0, method="BFGS")

        return opt
