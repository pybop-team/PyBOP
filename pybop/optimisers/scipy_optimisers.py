from collections.abc import Callable
from dataclasses import dataclass
from time import time

import numpy as np
from scipy.optimize import Bounds, OptimizeResult, differential_evolution, minimize

import pybop
from pybop import (
    BaseOptimiser,
    OptimisationResult,
    PopulationEvaluator,
    ScalarEvaluator,
)
from pybop._logging import Logger
from pybop.problems.base_problem import Problem

__all__: list[str] = [
    "SciPyMinimize",
    "SciPyDifferentialEvolution",
]


class BaseSciPyOptimiser(BaseOptimiser):
    """
    A base class for defining optimisation methods from the SciPy library.

    Parameters
    ----------
    problem : pybop.Problem
        The problem to optimise.
    options : pybop.OptimiserOptions
        Valid SciPy option keys and their values.
    """

    def __init__(
        self,
        problem: Problem,
        options: pybop.OptimiserOptions | None,
    ):
        super().__init__(problem, options=options)

    def scipy_bounds(self) -> Bounds:
        bounds = self.problem.parameters.get_bounds(transformed=True)
        # Convert bounds to SciPy format
        if isinstance(bounds, dict):
            return Bounds(bounds["lower"], bounds["upper"], True)
        elif isinstance(bounds, Bounds) or bounds is None:
            return bounds
        else:
            raise TypeError(
                "Bounds provided must be either type dict or SciPy.optimize.bounds object."
            )

    def _set_callback(self):
        """Add a callback to record the iteration number."""

        def base_callback(intermediate_result: OptimizeResult | np.ndarray):
            """
            Update the iteration number. Depending on the optimisation method, intermediate_result
            may be either an OptimizeResult or an array of the current parameter values.
            """
            self._logger.iteration += 1

        user_callback = self._options_dict.pop("callback", None)
        if user_callback is not None:
            self._options_dict["callback"] = (
                lambda x: user_callback(base_callback(x))
                if self._options_dict.get("method", None) != "trust-constr"
                else lambda x, intermediate_result: user_callback(
                    base_callback(intermediate_result)
                )
            )
        else:
            self._options_dict["callback"] = (
                base_callback
                if self._options_dict.get("method", None) != "trust-constr"
                else lambda x, intermediate_result: base_callback(intermediate_result)
            )


@dataclass
class SciPyMinimizeOptions(pybop.OptimiserOptions):
    """
    Options for the SciPy minimize method.

    Attributes
    ----------
    method : str, optional
        The optimisation method to use (default: None).
    jac : bool, optional
        Method for computing the gradient vector (default: None).
    tol : float, optional
        Tolerance for termination (default: None).
    maxiter : int, optional
        Maximum number of iterations (default: 1000).
    disp : bool, optional
        Set to True to print convergence messages (default: False).
    constraints : scipy constraint or list of scipy constraints, optional
        Constraints definition. Only for COBYLA, COBYQA, SLSQP and trust-constr.
    solver_options : dict, optional
        A dictionary of additional solver options, not including maxiter or disp.

    """

    method: str | None = None
    jac: bool | None = None
    tol: float | None = None
    maxiter: int = 1000
    disp: bool = False
    constraints: list | None = None
    solver_options: dict | None = None

    def validate(self):
        super().validate()
        if self.maxiter <= 0:
            raise ValueError("maxiter must be a positive integer.")
        if self.tol is not None and self.tol <= 0:
            raise ValueError("tol must be a positive float.")

    def to_dict(self) -> dict:
        """Convert the options to a dictionary format."""
        if self.solver_options is not None:
            solver_options = self.solver_options.copy()
        else:
            solver_options = {}
        for key in ["maxiter", "disp"]:
            if getattr(self, key) is not None:
                solver_options[key] = getattr(self, key)

        ret = {"options": solver_options}
        for key in ["method", "jac", "tol", "constraints"]:
            if getattr(self, key) is not None:
                ret[key] = getattr(self, key)
        return ret


class SciPyMinimize(BaseSciPyOptimiser):
    """
    Adapter for various scalar minimisation algorithms implemented in SciPy, allowing fine-tuning
    of the optimisation process through method selection and option configuration.

    Parameters
    ----------
    problem : pybop.Problem
        The problem to optimise.
    options: ScipyMinizeOptions, optional
        Options for the SciPy minimize method (default: None).

    See Also
    --------
    scipy.optimize.minimize : The SciPy method this class is based on.

    Notes
    -----
    Different optimisation methods may support different options. Consult SciPy's
    documentation for method-specific options and constraints.
    """

    def __init__(
        self,
        problem: Problem,
        options: SciPyMinimizeOptions | None = None,
    ):
        options = options or self.default_options()
        super().__init__(problem=problem, options=options)

    @staticmethod
    def default_options() -> SciPyMinimizeOptions:
        """Returns the default options for the optimiser."""
        return SciPyMinimizeOptions()

    def _set_up_optimiser(self):
        """
        Parse optimiser options.
        """
        self._options_dict = self._options.to_dict()

        self._cost0 = np.abs(self.problem.get_finite_initial_cost())
        self._x0 = self.problem.parameters.get_initial_values(transformed=True)
        self._options_dict["x0"] = self._x0
        self._options_dict["bounds"] = self.scipy_bounds()

        # If the problem has sensitivities, enable the Jacobian by default
        if self._options_dict.get("jac", None) is None:
            self._options_dict["jac"] = self.problem.has_sensitivities
        self._needs_sensitivities = True if self._options_dict["jac"] else False

        # Create logger and evaluator objects
        self._logger = Logger(
            minimising=self.problem.minimising,
            verbose=self.verbose,
            verbose_print_rate=self.verbose_print_rate,
        )
        self._evaluator = ScalarEvaluator(
            problem=self._problem,
            minimise=True,
            with_sensitivities=self._needs_sensitivities,
            logger=self._logger,
        )
        self._set_callback()

    def cost_wrapper(self, x):
        """
        Scale the cost function, preserving the sign, and eliminate nan values.
        """
        if not self._needs_sensitivities:
            cost = self._evaluator.evaluate(x)
            scaled_cost = cost / self._cost0
            if np.isinf(scaled_cost):
                self.inf_count += 1
                scaled_cost = np.sign(cost) * (
                    1 + 0.9**self.inf_count
                )  # for fake finite gradient
            return scaled_cost

        L, dl = self._evaluator.evaluate(x)
        return (L[0] / self._cost0, dl / self._cost0)

    def _run(self):
        """
        Executes the optimisation process using SciPy's minimize function.

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            The result of the optimisation including the optimised parameter values and cost.
        """
        start_time = time()

        # Set counters
        self.inf_count = 0
        self._logger.iteration = 1

        result: OptimizeResult = minimize(fun=self.cost_wrapper, **self._options_dict)
        if hasattr(result, "nit"):
            # Subtract final callback depending on method
            self._logger.iteration = result.nit

        total_time = time() - start_time

        # Log the optimised result as the final evaluation
        self._evaluator.evaluate(result.x)

        return OptimisationResult(
            optim=self,
            logger=self._logger,
            time=total_time,
            optim_name=self.name,
            message=result.message,
            scipy_result=result,
        )

    @property
    def name(self):
        """Provides the name of the optimisation strategy."""
        return "SciPyMinimize"


@dataclass
class SciPyDifferentialEvolutionOptions(pybop.OptimiserOptions):
    """
    Options for the SciPy differential evolution method.

    Parameters
    ----------
    strategy : str, optional
        The differential evolution strategy to use. Should be one of:
        - 'best1bin'
        - 'best1exp'
        - 'rand1exp'
        - 'randtobest1exp'
        - 'currenttobest1exp'
        - 'best2exp'
        - 'rand2exp'
        - 'randtobest1bin'
        - 'currenttobest1bin'
        - 'best2bin'
        - 'rand2bin'
        - 'rand1bin'
        Default is 'best1bin'.
    maxiter : int, optional
        Maximum number of generations (default: 1000).
    popsize : int, optional
        Multiplier for setting the total population size. The population has
        popsize * len(x) individuals (default: 15).
    tol : float, optional
        Relative tolerance for convergence (default: 0.01).
    mutation : float or tuple(float, float), optional
        The mutation constant. If specified as a float, should be in [0, 2].
        If specified as a tuple (min, max), dithering is used (default: (0.5, 1.0)).
    recombination : float, optional
        The recombination constant, should be in [0, 1] (default: 0.7).
    seed : int, optional
        Random seed for reproducibility.
    disp : bool, optional
        Display status messages (default: False).
    callback : Callable, optional
        Called after each iteration with the current result as argument.
    polish : bool, optional
        If True, performs a local optimisation on the solution (default: True).
    init : str or array-like, optional
        Specify initial population. Can be 'latinhypercube', 'random',
        or an array of shape (M, len(x)).
    atol : float, optional
        Absolute tolerance for convergence (default: 0).
    updating : {'immediate', 'deferred'}, optional
        If 'immediate', best solution vector is continuously updated within
        a single generation (default: 'immediate').
    workers : int or map-like Callable, optional
        If workers is an int the population is subdivided into workers
        sections and evaluated in parallel (default: 1).
    constraints : {NonlinearConstraint, LinearConstraint, Bounds}, optional
        Constraints on the solver.
    """

    strategy: str = "best1bin"
    maxiter: int = 1000
    tol: float = 0.01
    popsize: int | None = None
    mutation: float | tuple | None = None
    recombination: float | None = None
    seed: int | None = None
    disp: bool | None = None
    callback: Callable | None = None
    polish: bool | None = None
    init: str | np.ndarray | None = None
    atol: float | None = None

    def to_dict(self) -> dict:
        """Convert the options to a dictionary format."""
        ret = {
            "strategy": self.strategy,
            "maxiter": self.maxiter,
            "tol": self.tol,
        }
        optional_keys = [
            "popsize",
            "mutation",
            "recombination",
            "seed",
            "disp",
            "callback",
            "polish",
            "init",
            "atol",
        ]
        for key in optional_keys:
            if getattr(self, key) is not None:
                ret[key] = getattr(self, key)
        return ret


class SciPyDifferentialEvolution(BaseSciPyOptimiser):
    """
    Adapter for SciPy's differential_evolution function for global optimisation, useful for problems
    involving continuous parameters and potentially multiple local minima.

    Parameters
    ----------
    problem : pybop.Problem
        The problem to optimise.
    options: SciPyDifferentialEvolutionOptions, optional
        Options for the SciPy differential evolution method (default: None).

    See Also
    --------
    scipy.optimize.differential_evolution : The SciPy method this class is based on.

    Notes
    -----
    Differential Evolution is a stochastic population based method that is useful for
    global optimisation problems. At each pass through the population the algorithm mutates
    each candidate solution by mixing with other candidate solutions to create a trial
    candidate. The fitness of all candidates is then evaluated and for each candidate if
    the trial candidate is an improvement, it takes its place in the population for the next
    iteration.
    """

    def __init__(
        self,
        problem: Problem,
        options: SciPyDifferentialEvolutionOptions | None = None,
    ):
        options = options or self.default_options()
        super().__init__(problem=problem, options=options)

    @staticmethod
    def default_options() -> SciPyDifferentialEvolutionOptions:
        """Returns the default options for the optimiser."""
        return SciPyDifferentialEvolutionOptions()

    def _set_up_optimiser(self):
        """
        Parse optimiser options.
        """
        self._options_dict = self._options.to_dict()
        self._needs_sensitivities = False

        # Check bounds
        bounds = self.scipy_bounds()
        if bounds is None:
            raise ValueError("Bounds must be specified for differential_evolution.")
        else:
            if not (np.isfinite(bounds.lb).all() and np.isfinite(bounds.ub).all()):
                raise ValueError("Bounds must be finite for differential_evolution.")
        self._options_dict["bounds"] = bounds

        # Create logger and evaluator objects
        self._logger = Logger(
            minimising=self.problem.minimising,
            verbose=self.verbose,
            verbose_print_rate=self.verbose_print_rate,
        )
        self._evaluator = ScalarEvaluator(
            problem=self._problem,
            minimise=True,
            with_sensitivities=self._needs_sensitivities,
            logger=self._logger,
        )
        self._set_callback()

        # Enable vectorisation. Differential evolution proposes candidates as an
        # array of size (N, S) amd expects to receive a set of costs of size (S,)
        self._options_dict["updating"] = "deferred"
        pop_evaluator = PopulationEvaluator(
            problem=self._problem,
            minimise=True,
            with_sensitivities=self._needs_sensitivities,
            logger=self._logger,
        )

        def map_function(func, positions):
            # Use the PopulationEvaluator instead of the ScalarEvaluator for multiprocessing
            return pop_evaluator.evaluate(positions)

        self._options_dict["workers"] = map_function

    def _run(self):
        """
        Executes the optimisation process using SciPy's differential_evolution function.

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            The result of the optimisation including the optimised parameter values and cost.
        """
        start_time = time()

        # Set counters
        self._logger.iteration = 1

        result = differential_evolution(
            func=self._evaluator.evaluate, **self._options_dict
        )
        self._logger.iteration -= 1  # undo the final callback

        total_time = time() - start_time

        # Log the optimised result as the final evaluation
        self._evaluator.evaluate(result.x)

        return OptimisationResult(
            optim=self,
            logger=self._logger,
            time=total_time,
            optim_name=self.name,
            message=result.message,
            scipy_result=result,
        )

    @property
    def name(self):
        """Provides the name of the optimisation strategy."""
        return "SciPyDifferentialEvolution"
