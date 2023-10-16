import pybop
import nlopt
from pybop.optimisers.base_optimiser import BaseOptimiser


class NLoptOptimize(BaseOptimiser):
    """
    Wrapper class for the NLOpt optimiser class. Extends the BaseOptimiser class.
    """

    def __init__(self, method=None, x0=None, xtol=None):
        super().__init__()
        self.name = "NLOpt Optimiser"

        if method is not None:
            self.opt = nlopt.opt(method, len(x0))
        else:
            self.opt = nlopt.opt(nlopt.LN_BOBYQA, len(x0))

        if xtol is not None:
            self.opt.set_xtol_rel(xtol)
        else:
            self.opt.set_xtol_rel(1e-5)

    def _runoptimise(self, cost_function, x0, bounds):
        """
        Run the NLOpt optimisation method.

        Parameters
        ----------
        cost_function: function for optimising
        method: optimisation algorithm
        x0: initialisation array
        bounds: bounds array
        """

        self.opt.set_min_objective(cost_function)
        self.opt.set_lower_bounds(bounds["lower"])
        self.opt.set_upper_bounds(bounds["upper"])
        results = self.opt.optimize(x0)
        num_evals = self.opt.get_numevals()

        return results, self.opt.last_optimum_value(), num_evals
