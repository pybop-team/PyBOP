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

    def check_monotonicity(self, voltage: np.ndarray) -> None:
        """
        Check if voltage data is monotonic and warn if not.

        Parameters
        ----------
        voltage : np.ndarray
            Voltage array to check for monotonicity.
        """
        is_increasing = np.all(np.diff(voltage) > 0)
        is_decreasing = np.all(np.diff(voltage) < 0)

        if not (is_increasing or is_decreasing):
            warnings.warn("OCV is not strictly monotonic.", stacklevel=2)


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
        bounds_error: bool = False,
        fill_value: str | float = "extrapolate",
        axis: int = 0,
    ):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.name = name
        self._interp_func = self._create_interpolant(bounds_error, fill_value, axis)

    def _create_interpolant(
        self, bounds_error: bool, fill_value: str | float, axis: int
    ):
        """Create the scipy interpolation function."""
        return interpolate.interp1d(
            self.x,
            self.y,
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
        float, array_like, or PybammInterpolant
            Interpolated values or PyBaMM interpolant object.
        """
        try:
            # Try numeric evaluation first
            return self._interp_func(x)
        except Exception:
            # Fall back to PyBaMM interpolant for symbolic evaluation
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
    optimiser_options : pybop.OptimiserOptions, optional
        Options for the optimiser.
    verbose : bool, optional
        If True, progress messages are printed (default: False).
    """

    def __init__(
        self,
        ocv_function: Callable,
        optimiser: pybop.BaseOptimiser | None = None,
        optimiser_options: pybop.OptimiserOptions | None = None,
        verbose: bool = False,
    ):
        self.ocv_function = ocv_function
        self.optimiser = optimiser or pybop.SciPyMinimize
        self.optimiser_options = optimiser_options
        self.verbose = verbose

    def _create_root_function(self, ocv_value: float) -> Callable:
        """
        Create the root-finding objective function.

        Parameters
        ----------
        ocv_value : float
            Target OCV value.

        Returns
        -------
        Callable
            Objective function for optimisation.
        """

        def ocv_root(x, **kwargs):
            return np.abs(self.ocv_function(x[0]) - ocv_value)

        return ocv_root

    def _build_problem(self, ocv_root_func: Callable) -> pybop.Problem:
        """
        Build the problem for root finding.

        Parameters
        ----------
        ocv_root_func : Callable
            Root-finding objective function.

        Returns
        -------
        pybop.Problem
            Configured optimisation problem.
        """
        builder = pybop.builders.Python()
        builder.add_parameter(pybop.Parameter("OCV root", initial_value=0.5))
        builder.add_fun(ocv_root_func)
        return builder.build()

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
        # Create objective function and optimisation problem
        ocv_root_func = self._create_root_function(ocv_value)
        problem = self._build_problem(ocv_root_func)

        # Set default optimiser options if not provided
        if self.optimiser_options is None:
            self.optimiser_options = pybop.ScipyMinimizeOptions(verbose=self.verbose)

        # Run optimisation
        optim = self.optimiser(problem, options=self.optimiser_options)
        results = optim.run()

        return results.x[0]
