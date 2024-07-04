import numpy as np
from scipy.optimize import minimize

from pybop import BaseOptimiser, Result


class StandaloneOptimiser(BaseOptimiser):
    """
    Defines an example standalone optimiser without a Cost.
    """

    def __init__(self, cost=None, **optimiser_kwargs):
        # Define cost function
        def cost(x):
            x1, x2 = x
            return (x1 - 2) ** 2 + (x2 - 4) ** 4

        # Set initial values and other options
        optimiser_options = dict(
            x0=np.array([0, 0]),
            bounds=None,
            method="Nelder-Mead",
            jac=False,
            maxiter=100,
        )
        optimiser_options.update(optimiser_kwargs)
        super().__init__(cost, **optimiser_options)

    def _set_up_optimiser(self):
        """
        Parse optimiser options.
        """
        # Reformat bounds
        if isinstance(self.bounds, dict):
            self._scipy_bounds = [
                (lower, upper)
                for lower, upper in zip(self.bounds["lower"], self.bounds["upper"])
            ]
        else:
            self._scipy_bounds = self.bounds

        # Parse additional options and remove them from the options dictionary
        self._options = self.unset_options
        self.unset_options = dict()
        self._options["options"] = self._options.pop("options", dict())
        if "maxiter" in self._options.keys():
            # Nest this option within an options dictionary for SciPy minimize
            self._options["options"]["maxiter"] = self._options.pop("maxiter")

    def _run(self):
        """
        Executes the optimisation process using SciPy's minimize function.

        Returns
        -------
        x : numpy.ndarray
            The best parameter set found by the optimization.
        final_cost : float
            The final cost associated with the best parameters.
        """
        self.log = [[self.x0]]

        # Add callback storing history of parameter values
        def callback(x):
            self.log.append([x])

        # Run optimiser
        result = minimize(
            self.cost,
            self.x0,
            bounds=self._scipy_bounds,
            callback=callback,
            **self._options,
        )

        return Result(
            x=result.x,
            final_cost=self.cost(result.x),
            n_iterations=result.nit,
            scipy_result=result,
        )

    def name(self):
        """
        Provides the name of the optimization strategy.

        Returns
        -------
        str
            The name 'SciPyMinimize'.
        """
        return "StandaloneOptimiser"
