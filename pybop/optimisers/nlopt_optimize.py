import nlopt
from .base_optimiser import BaseOptimiser
import numpy as np


class NLoptOptimize(BaseOptimiser):
    """
    Extends BaseOptimiser to utilize the NLopt library for nonlinear optimization.

    This class serves as an interface to the NLopt optimization algorithms. It allows the user to
    define an optimization problem with bounds, initial guesses, and to select an optimization method
    provided by NLopt.

    Parameters
    ----------
    n_param : int
        Number of parameters to optimize.
    xtol : float, optional
        The relative tolerance for optimization (stopping criteria). If not provided, a default of 1e-5 is used.
    method : nlopt.algorithm, optional
        The NLopt algorithm to use for optimization. If not provided, LN_BOBYQA is used by default.
    maxiter : int, optional
        The maximum number of iterations to perform during optimization. If not provided, NLopt's default is used.
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
        Runs the optimization process using the NLopt library.

        Parameters
        ----------
        cost_function : callable
            The objective function to minimize. It should take an array of parameter values and return the scalar cost.
        x0 : array_like
            The initial guess for the parameters.
        bounds : dict
            A dictionary containing the 'lower' and 'upper' bounds arrays for the parameters.

        Returns
        -------
        tuple
            A tuple containing the optimized parameter values and the final cost.
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
        Indicates if the optimiser requires gradient information for the cost function.

        Returns
        -------
        bool
            False, as the default NLopt algorithms do not require gradient information.
        """
        return False

    def name(self):
        """
        Returns the name of this optimiser instance.

        Returns
        -------
        str
            The name 'NLoptOptimize' representing this NLopt optimization class.
        """
        return "NLoptOptimize"
