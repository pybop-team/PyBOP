import pybop
import nlopt


class nlopt_opt(pybop.BaseOptimisation):
    """
    Wrapper class for the NLOpt optimisation class. Extends the BaseOptimisation class.
    """

    def __init__(self, cost_function, x0, bounds, method=None, options=None):
        super().__init__()
        self.cost_function = cost_function
        self.method = method
        self.x0 = x0
        self.bounds = bounds
        self.options = options
        self.name = "NLOpt Optimisation"

    def _runoptimise(self):
        """
        Run the NLOpt opt method.

        Parameters
        ----------
        cost_function: function for optimising
        method: optimisation method
        x0: Initialisation array
        options: options dictionary
        bounds: bounds array
        """
        if self.options.xtol is not None:
            opt.set_xtol_rel(self.options.xtol)

        if self.method is not None:
            opt = nlopt.opt(self.method, len(self.x0))
        else:
            opt = nlopt.opt(nlopt.LN_BOBYQA, len(self.x0))

        opt.set_min_objective(self.cost_function)
        opt.set_lower_bounds(self.bounds.lower)
        opt.set_upper_bounds(self.bounds.upper)
        results = opt.optimize(self.cost_function)
        num_evals = opt.get_numevals()

        return results, opt.last_optimum_value(), num_evals
