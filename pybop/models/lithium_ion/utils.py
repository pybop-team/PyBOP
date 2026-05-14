from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from pybamm import Interpolant as PybammInterpolant
from pybamm import Symbol
from scipy import interpolate

if TYPE_CHECKING:
    from pybop.optimisers.base_optimiser import BaseOptimiser, OptimiserOptions

from pybop.costs.design_cost import DesignCost
from pybop.optimisers.scipy_optimisers import SciPyMinimize
from pybop.parameters.parameter import Parameter, Parameters
from pybop.problems.problem import Problem
from pybop.simulators.base_simulator import BaseSimulator
from pybop.simulators.solution import Solution


class Interpolant:
    """
    A class that returns a pybamm.Interpolant to pybamm models and otherwise
    a numeric interpolant.

    Parameters
    ----------
    x : array_like
        Input coordinates.
    y : array_like
        Output values corresponding to x.
    name : str, optional
        Name for the interpolant when used in PyBaMM.
    kind: str, optional
        Which kind of interpolator to use. Can be "linear" (default) or "cubic".
    bounds_error : bool, optional
        If True, raise error when interpolating outside bounds.
    fill_value : str or float, optional
        Value to use for out-of-bounds interpolation.
    axis : int, optional
        Axis along which to interpolate.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        name: str | None = None,
        kind: str | None = None,
        bounds_error: bool = False,
        fill_value: str | float = "extrapolate",
        axis: int = 0,
    ):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.name = name if name is None else name.replace(" ", "_")
        self.kind = kind or "linear"
        self._interp_func = self._create_interpolant(bounds_error, fill_value, axis)

    def _create_interpolant(
        self, bounds_error: bool, fill_value: str | float, axis: int
    ):
        """Create the scipy interpolation function."""
        return interpolate.interp1d(
            self.x,
            self.y,
            kind=self.kind,
            bounds_error=bounds_error,
            fill_value=fill_value,
            axis=axis,
        )

    def __call__(self, x: float | np.ndarray):
        """
        Evaluate the interpolant at given points.

        Parameters
        ----------
        x : float or array_like
            Points at which to evaluate the interpolant.

        Returns
        -------
        float, array_like, or pybamm.Interpolant
            Interpolated values or PyBaMM interpolant object.
        """
        if isinstance(x, Symbol):  # symbolic evaluation
            return PybammInterpolant(
                self.x, self.y, x, name=self.name, interpolator=self.kind
            )
        else:  # numeric evaluation
            return self._interp_func(x)

    def __repr__(self):
        return f"{self.name}, {self.kind} interpolant"


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
    optimiser_options : pybop.OptimiserOptions, optional
        Options for the optimiser.
    """

    def __init__(
        self,
        ocv_function: Callable,
        optimiser: "BaseOptimiser | None" = None,
        optimiser_options: "OptimiserOptions | None" = None,
    ):
        self.optimiser = optimiser or SciPyMinimize
        self.optimiser_options = optimiser_options or self.optimiser.default_options()

        self.parameters = Parameters(
            {"Root": Parameter(initial_value=0.5, bounds=[0, 1])}
        )

        # Set up a root-finding cost function
        class OCVRoot(BaseSimulator):
            def __init__(self, parameters: Parameters, ocv_value: float):
                super().__init__(parameters=parameters)
                self.ocv_value = ocv_value

            def solve_batch(self, inputs, calculate_sensitivities: bool = False):
                solutions = []
                for x in inputs:
                    diff = np.abs(ocv_function(x["Root"]) - self.ocv_value)
                    sol = Solution()
                    sol.set_solution_variable("Difference", data=np.asarray([diff]))
                    solutions.append(sol)
                return solutions

        self.ocv_root = OCVRoot

        # Minimise to find the stoichiometry
        self.cost = DesignCost(target="Difference")
        self.cost.minimising = True

    def __call__(self, ocv_value: float) -> float:
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
        problem = Problem(self.ocv_root(self.parameters, ocv_value), self.cost)
        optim = self.optimiser(problem, options=self.optimiser_options)
        result = optim.run()
        return result.best_inputs["Root"]
