import nlopt
from .base_optimiser import BaseOptimiser


class NLoptOptimize(BaseOptimiser):
    """
    Wrapper class for the NLOpt optimiser class. Extends the BaseOptimiser class.
    """

    def __init__(self, n_param, xtol=None, method=None):
        super().__init__()
        self.name = "NLoptOptimize"

        if method is not None:
            self.optim = nlopt.opt(method, n_param)
        else:
            self.optim = nlopt.opt(nlopt.LN_BOBYQA, n_param)

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

        # Pass settings to the optimiser
        self.optim.set_min_objective(cost_function.compute)
        self.optim.set_lower_bounds(bounds["lower"])
        self.optim.set_upper_bounds(bounds["upper"])

        # Run the optimser
        x = self.optim.optimize(x0)

        # Get performance statistics
        output = self.optim
        final_cost = self.optim.last_optimum_value()
        num_evals = self.optim.get_numevals()

        return x, output, final_cost, num_evals
