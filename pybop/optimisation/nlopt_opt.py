import pybop
import nlopt


class nlopt_opt(pybop.BaseOptimisation):
    """
    Wrapper class for the NLOpt optimisation class. Extends the BaseOptimisation class.
    """

    def __init__(self, cost_function, method, x0, bounds, options):
        super().__init__()
        self.cost_function = cost_function
        self.method = method
        self.x0 = x0 or cost_function.x0
        self.bounds = bounds or cost_function.bounds
        self.options = options
        self.name = "NLOpt Optimisation"

    def _runoptimise(cost_function, method, x0, bounds, options):
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
        if options.xtol != None:
            opt.set_xtol_rel(options.xtol)

        if method != None:
            opt = nlopt.opt(method, len(x0))
        else:
            opt = nlopt.opt(nlopt.LN_BOBYQA, len(x0))

        opt.set_min_objective(cost_function)

        opt.set_lower_bounds(bounds.lower)
        opt.set_upper_bounds(bounds.upper)
        results = opt.optimize(cost_function)
        num_evals = opt.get_numevals()

        return results, opt.last_optimum_value(), num_evals
