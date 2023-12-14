import nlopt
from .base_optimiser import BaseOptimiser
import numpy as np


class NLoptOptimize(BaseOptimiser):
    """
    Wrapper class for the NLOpt optimiser class. Extends the BaseOptimiser class.
    """

    def __init__(self, n_param, xtol=None, method=None, maxiter=None):
        super().__init__()
        self.n_param = n_param
        self._max_iterations = maxiter

        if method is not None:
            self.optim = nlopt.opt(method, self.n_param)
        else:
            self.optim = nlopt.opt(nlopt.LN_BOBYQA, self.n_param)

        if xtol is not None:
            self.optim.set_xtol_rel(xtol)
        else:
            self.optim.set_xtol_rel(1e-5)

    def _runoptimise(self, cost_function, x0, bounds):
        """
        Run the NLOpt optimisation method.

        Inputs
        ----------
        cost_function: function for optimising
        method: optimisation algorithm
        x0: initialisation array
        bounds: bounds array
        """

        # Add callback storing history of parameter values
        self.log = [[x0]]

        def cost_wrapper(x, grad):
            self.log.append([np.array(x)])
            return cost_function(x, grad)

        # Pass settings to the optimiser
        self.optim.set_min_objective(cost_wrapper)
        self.optim.set_lower_bounds(bounds["lower"])
        self.optim.set_upper_bounds(bounds["upper"])

        # Set max iterations
        if self._max_iterations is not None:
            self.optim.set_maxeval(self._max_iterations)

        # Run the optimser
        x = self.optim.optimize(x0)

        # Get performance statistics
        final_cost = self.optim.last_optimum_value()

        return x, final_cost

    def needs_sensitivities(self):
        """
        Returns True if the optimiser needs sensitivities.
        """
        return False

    def name(self):
        """
        Returns the name of the optimiser.
        """
        return "NLoptOptimize"
