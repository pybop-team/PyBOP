import warnings
from collections.abc import Callable

import numpy as np
from pybamm import Interpolant as PybammInterpolant
from scipy import interpolate

import pybop


class BaseApplication:
    """
    A base class for PyBOP's application methods.
    """

    def check_monotonicity(self, voltage):
        if not (
            all(x < y for x, y in zip(voltage, voltage[1:], strict=False))
            or all(x > y for x, y in zip(voltage, voltage[1:], strict=False))
        ):
            warnings.warn("OCV is not strictly monotonic.", stacklevel=1)


class Interpolant:
    """
    A class that returns a pybamm.Interpolant to pybamm models and otherwise
    a numeric interpolant.
    """

    def __init__(
        self, x, y, name=None, bounds_error=False, fill_value="extrapolate", axis=0
    ):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.name = name
        self.interp1d = interpolate.interp1d(
            x,
            y,
            bounds_error=bounds_error,
            fill_value=fill_value,
            axis=axis,
        )

    def __call__(self, x):
        try:
            # Try to evaluate the interpolant numerically, this will return an
            # error if x is a PyBaMM object
            return self.interp1d(x)
        except (Exception, SystemExit, KeyboardInterrupt):
            # Evaluate the interpolant as a PyBaMM function for use in a model
            return PybammInterpolant(self.x, self.y, x, name=self.name)


class InverseOCV:
    """
    A class to find the stoichiometry corresponding to a given open-circuit
    voltage.

    Parameters
    ----------
    ocv_function : Callable
        The open-circuit voltage as a function of stoichiometry.
    optimiser : pybop.BaseOptimiser, optional
        The optimisation algorithm to use (default: pybop.SciPyMinimize).
    verbose : bool, optional
        If True, progress messages are printed (default: False).
    """

    def __init__(
        self,
        ocv_function: Callable,
        optimiser: pybop.BaseOptimiser | None = pybop.SciPyMinimize,
        verbose: bool = False,
    ):
        self.ocv_function = ocv_function
        self.optimiser = optimiser
        self.verbose = verbose

    def __call__(self, ocv_value: float):
        """
        Estimate and return the stoichiometry.

        Parameters
        ----------
        ocv_value : float
            The open-circuit voltage value [V] for which to estimate the stoichiometry.

        Returns
        -------
        float
            The stoichiometry corresponding to the open-circuit voltage value.
        """
        ocv_function = self.ocv_function

        self.parameters = pybop.Parameters(
            {"Root": pybop.ParameterBounds(initial_value=0.5, bounds=[0, 1])}
        )

        # Set up a root-finding cost function
        class OCVRoot(pybop.BaseSimulator):
            def batch_solve(self, inputs, calculate_sensitivities: bool = False):
                solutions = []
                for x in inputs:
                    diff = np.abs(ocv_function(x["Root"]) - ocv_value)
                    sol = pybop.Solution()
                    sol.set_solution_variable("Difference", data=np.asarray([diff]))
                    solutions.append(sol)
                return solutions

        # Minimise to find the stoichiometry
        cost = pybop.DesignCost(target="Difference")
        cost.minimising = True
        problem = pybop.Problem(OCVRoot(self.parameters), cost)
        options = pybop.SciPyMinimizeOptions(verbose=self.verbose)
        optim = self.optimiser(problem=problem, options=options)
        result = optim.run()
        return result.best_inputs["Root"]
