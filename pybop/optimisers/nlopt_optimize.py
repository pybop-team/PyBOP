import pybop
import nlopt
from .base_optimiser import BaseOptimiser


class NLoptOptimize(BaseOptimiser):
    """
    Wrapper class for the NLOpt optimiser class. Extends the BaseOptimiser class.
    """

    def __init__(self, method=None, x0=None, xtol=None):
        super().__init__()
        self.name = "NLOpt Optimiser"

        if method is not None:
            self.optim = nlopt.opt(method, len(x0))
        else:
            self.optim = nlopt.opt(nlopt.LN_BOBYQA, len(x0))

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

        self.optim.set_min_objective(cost_function)
        self.optim.set_lower_bounds(bounds["lower"])
        self.optim.set_upper_bounds(bounds["upper"])

        # Run the optimser
        x = self.optim.optimize(x0)

        # Get performance statistics
        output = self.optim
        final_cost = self.optim.last_optimum_value()
        num_evals = self.optim.get_numevals()

        return x, output, final_cost, num_evals
