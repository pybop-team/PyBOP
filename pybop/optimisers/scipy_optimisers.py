from collections.abc import Callable
from dataclasses import dataclass
from time import time

import numpy as np
from scipy.optimize import Bounds, OptimizeResult, differential_evolution, minimize

import pybop
from pybop import BaseOptimiser, OptimisationResult, SciPyEvaluator
from pybop.problems.base_problem import Problem


class BaseSciPyOptimiser(BaseOptimiser):
    """
    A base class for defining optimisation methods from the SciPy library.

    Parameters
    ----------
    x0 : array_like
        Initial position from which optimisation will start.
    bounds : dict, sequence or scipy.optimize.Bounds, optional
        Bounds for variables as supported by the selected method.
    **optimiser_kwargs : optional
        Valid SciPy option keys and their values.
    """

    def __init__(
        self,
        problem: Problem,
        needs_sensitivities: bool,
        options: pybop.OptimiserOptions,
    ):
        self._needs_sensitivities = needs_sensitivities
        self._intermediate_x_search = []
        self._intermediate_x_model = []
        self._intermediate_cost = []
        super().__init__(problem, options=options)

    def store_intermediate_results(
        self,
        x_search: np.ndarray,
        x_model: np.ndarray,
        cost: np.ndarray,
    ):
        self._intermediate_x_search.append(x_search)
        self._intermediate_x_model.append(x_model)
        self._intermediate_cost.extend(cost.tolist())

    def get_and_reset_itermediate_results(self):
        ret = (
            self._intermediate_x_search,
            self._intermediate_x_model,
            self._intermediate_cost,
        )
        self._intermediate_x_search = []
        self._intermediate_x_model = []
        self._intermediate_cost = []
        return ret

    def scipy_bounds(self) -> Bounds:
        bounds = self.problem.params.get_bounds()
        # Convert bounds to SciPy format
        if isinstance(bounds, dict):
            return Bounds(bounds["lower"], bounds["upper"], True)
        elif isinstance(bounds, Bounds) or bounds is None:
            return bounds
        else:
            raise TypeError(
                "Bounds provided must be either type dict or SciPy.optimize.bounds object."
            )

    def needs_sensitivities(self) -> bool:
        """
        Returns whether the optimiser needs sensitivities.

        Returns
        -------
        bool
            True if the optimiser needs sensitivities, False otherwise.
        """
        return self._needs_sensitivities

    def evaluator(self) -> SciPyEvaluator:
        """
        Internal method to run the optimisation using a PyBOP optimiser.

        Returns
        -------
        result : pybop.Result
            The result of the optimisation including the optimised parameter values and cost.
        """

        # Choose method to evaluate
        if self._needs_sensitivities:

            def fun(x):
                self.problem.set_params(x)
                return self.problem.run_with_sensitivities()

        else:

            def fun(x):
                self.problem.set_params(x)
                return self.problem.run()

        # Create evaluator object
        return SciPyEvaluator(fun)


@dataclass
class ScipyMinimizeOptions(pybop.OptimiserOptions):
    """
    Options for the SciPy minimization method.

    Attributes:
        method : str
            The optimisation method to use. Default is 'Nelder-Mead'.
        jac : bool
            Method for computing the gradient vector. Default is False.
        tol : float, optional
            Tolerance for termination. Default is None.
        maxiter : int
            Maximum number of iterations. Default is 1000.
        disp : bool, optional
            Set to True to print convergence messages. Default is False.
        ftol : float, optional
            Function tolerance for termination. Default is None.
        gtol : float, optional
            Gradient tolerance for termination. Default is None.
        eps : float, optional
            Step size for finite difference approximation. Default is None.
        maxcor : int, optional
            Maximum number of variable metric corrections. Default is None.
        maxfev : int, optional
            Maximum number of function evaluations. Default is None.

    """

    method: str = "Nelder-Mead"
    jac: bool = False
    tol: float | None = None
    maxiter: int = 1000
    disp: bool | None = False
    ftol: float | None = None
    gtol: float | None = None
    eps: float | None = None
    maxcor: int | None = None
    maxfev: int | None = None

    def validate(self):
        super().validate()
        if self.maxiter <= 0:
            raise ValueError("maxiter must be a positive integer.")
        if self.tol is not None and self.tol <= 0:
            raise ValueError("tol must be a positive float.")
        if self.eps is not None and self.eps <= 0:
            raise ValueError("eps must be a positive float.")
        if not isinstance(self.jac, bool):
            raise ValueError("jac must be a boolean value.")

    def to_dict(self) -> dict:
        """
        Convert the options to a dictionary format.

        Returns
        -------
        dict
            Dictionary representation of the options.
        """
        ret = {
            "method": self.method,
            "jac": self.jac,
            "options": {
                "maxiter": self.maxiter,
            },
        }
        if self.tol is not None:
            ret["tol"] = self.tol
        optional_keys = [
            "disp",
            "ftol",
            "gtol",
            "eps",
            "maxcor",
            "maxfev",
        ]
        for key in optional_keys:
            if getattr(self, key) is not None:
                ret["options"][key] = getattr(self, key)
        return ret


class SciPyMinimize(BaseSciPyOptimiser):
    """
    Adapts SciPy's minimize function for use as an optimisation strategy.

    This class provides an interface to various scalar minimisation algorithms implemented in SciPy,
    allowing fine-tuning of the optimisation process through method selection and option configuration.

    Parameters
    ----------
    problem : pybop.Problem
        The problem to be optimised.
    options: ScipyMinizeOptions, optional
        Options for the SciPy minimisation method. Default is None.

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
        options: ScipyMinimizeOptions | None = None,
    ):
        options = options or ScipyMinimizeOptions()
        self._options_dict = options.to_dict()
        self._iteration = 0
        super().__init__(
            problem=problem, options=options, needs_sensitivities=options.jac
        )
        self._evaluator = self.evaluator()
        self._cost0 = 1.0

    @staticmethod
    def default_options() -> ScipyMinimizeOptions:
        """
        Returns the default options for the optimiser.

        Returns
        -------
        ScipyMinimizeOptions
            The default options for the optimiser.
        """
        return ScipyMinimizeOptions()

    def _set_up_optimiser(self):
        """
        Parse optimiser options.
        """
        self._scipy_bounds = self.scipy_bounds()
        self._evaluator = self.evaluator()

    def cost_wrapper(self, x):
        """
        Scale the cost function, preserving the sign convention, and eliminate nan values
        """
        x_model = self.problem.params.transformation.to_model(x)
        if not self._options_dict["jac"]:
            cost = self._evaluator.evaluate(x_model)
            self.store_intermediate_results(x_search=x, x_model=x_model, cost=cost)
            scaled_cost = cost / self._cost0
            if np.isinf(scaled_cost):
                self.inf_count += 1
                scaled_cost = np.sign(cost) * (
                    1 + 0.9**self.inf_count
                )  # for fake finite gradient
            return scaled_cost

        L, dl = self._evaluator.evaluate(x_model)
        self.store_intermediate_results(x_search=x, x_model=x_model, cost=L)
        return (L[0] / self._cost0, dl / self._cost0)

    def _run(self):
        """
        Executes the optimisation process using SciPy's minimize function.

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            The result of the optimisation including the optimised parameter values and cost.
        """
        self.inf_count = 0
        self._iteration = 0

        # Add callback storing history of parameter values
        def base_callback(intermediate_result: OptimizeResult | np.ndarray):
            """
            Log intermediate optimisation solutions. Depending on the
            optimisation algorithm, intermediate_result may be either
            an OptimizeResult or an array of parameter values, with a
            try/except ensuring both cases are handled correctly.
            """
            if isinstance(intermediate_result, OptimizeResult):
                x_best = intermediate_result.x
                cost_best = intermediate_result.fun * self._cost0
            else:
                x_best = intermediate_result
                result = self._evaluator.evaluate(x_best)[0]
                cost_best = result[0] if self._needs_sensitivities else result
            print("cost_best:", cost_best)

            x_model_best = self.problem.params.transformation.to_model(x_best)

            (x_search, x_model, cost) = self.get_and_reset_itermediate_results()
            evaluations = len(x_search)
            self._iteration += 1

            self.logger.log_update(
                iterations=self._iteration,
                evaluations=evaluations,
                x_search_best=x_best,
                x_model_best=x_model_best,
                cost_best=cost_best,
                x_search=x_search,
                x_model=x_model,
                cost=cost,
            )

        callback = (
            base_callback
            if self._options_dict["method"] != "trust-constr"
            else lambda x, intermediate_result: base_callback(intermediate_result)
        )

        x0 = self.problem.params.get_initial_values(transformed=True)
        if self.needs_sensitivities():
            self._cost0 = self._evaluator.evaluate(x0)[0][0]
        else:
            self._cost0 = self._evaluator.evaluate(x0)[0]

        start_time = time()
        result: OptimizeResult = minimize(
            self.cost_wrapper,
            x0,
            bounds=self._scipy_bounds,
            callback=callback,
            **self._options_dict,
        )
        base_callback(result)  # Log final result
        total_time = time() - start_time

        try:
            nit = result.nit
        except AttributeError:
            nit = -1

        try:
            nfev = result.nfev
        except AttributeError:
            nfev = -1

        return OptimisationResult(
            best_cost=result.fun * self._cost0,
            n_iterations=nit,
            n_evaluations=nfev,
            problem=self.problem,
            x=result.x,
            time=total_time,
            message=result.message,
        )

    def name(self):
        """Provides the name of the optimisation strategy."""
        return "SciPyMinimize"


@dataclass
class SciPyDifferentialEvolutionOptions(pybop.OptimiserOptions):
    """
    Options for the SciPy differential evolution method.

    Attributes:
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
            Maximum number of generations. Default is 1000.
        popsize : int, optional
            Multiplier for setting the total population size. The population has
            popsize * len(x) individuals. Default is 15.
        tol : float, optional
            Relative tolerance for convergence. Default is 0.01.
        mutation : float or tuple(float, float), optional
            The mutation constant. If specified as a float, should be in [0, 2].
            If specified as a tuple (min, max), dithering is used. Default is (0.5, 1.0).
        recombination : float, optional
            The recombination constant, should be in [0, 1]. Default is 0.7.
        seed : int, optional
            Random seed for reproducibility.
        disp : bool, optional
            Display status messages. Default is False.
        callback : Callable, optional
            Called after each iteration with the current result as argument.
        polish : bool, optional
            If True, performs a local optimisation on the solution. Default is True.
        init : str or array-like, optional
            Specify initial population. Can be 'latinhypercube', 'random',
            or an array of shape (M, len(x)).
        atol : float, optional
            Absolute tolerance for convergence. Default is 0.
        updating : {'immediate', 'deferred'}, optional
            If 'immediate', best solution vector is continuously updated within
            a single generation. Default is 'immediate'.
        workers : int or map-like Callable, optional
            If workers is an int the population is subdivided into workers
            sections and evaluated in parallel. Default is 1.
        constraints : {NonlinearConstraint, LinearConstraint, Bounds}, optional
            Constraints on the solver.
    """

    strategy: str = "best1bin"
    max_iterations: int = 1000
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
    updating: str | None = None
    workers: int | Callable | None = None

    def to_dict(self) -> dict:
        """
        Convert the options to a dictionary format.

        Returns
        -------
        dict
            Dictionary representation of the options.
        """
        ret = {
            "strategy": self.strategy,
            "maxiter": self.max_iterations,
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
            "updating",
            "workers",
        ]
        for key in optional_keys:
            if getattr(self, key) is not None:
                ret[key] = getattr(self, key)
        return ret


class SciPyDifferentialEvolution(BaseSciPyOptimiser):
    """
    Adapts SciPy's differential_evolution function for global optimisation.

    This class provides a global optimisation strategy based on differential evolution, useful for
    problems involving continuous parameters and potentially multiple local minima.

    Parameters
    ----------
    problem : pybop.Problem
        The problem to be optimised.
    multistart : int, optional
        Number of independent runs of the optimisation algorithm. Default is 1.
    options: SciPyDifferentialEvolutionOptions, optional
        Options for the SciPy differential evolution method. Default is None.

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
        options = options or SciPyDifferentialEvolutionOptions()
        self._options_dict = options.to_dict()
        super().__init__(problem=problem, options=options, needs_sensitivities=False)
        self._evaluator = self.evaluator()
        self._iterations = 0

    @staticmethod
    def default_options() -> SciPyDifferentialEvolutionOptions:
        """
        Returns the default options for the optimiser.

        Returns
        -------
        SciPyDifferentialEvolutionOptions
            The default options for the optimiser.
        """
        return SciPyDifferentialEvolutionOptions()

    def _set_up_optimiser(self):
        """
        Parse optimiser options.
        """
        self._scipy_bounds = self.scipy_bounds()
        self._evaluator = self.evaluator()

        # Check bounds
        if self._scipy_bounds is None:
            raise ValueError("Bounds must be specified for differential_evolution.")
        else:
            bnds = self._scipy_bounds
            if not (np.isfinite(bnds.lb).all() and np.isfinite(bnds.ub).all()):
                raise ValueError("Bounds must be specified for differential_evolution.")

    def _run(self):
        """
        Executes the optimization process using SciPy's differential_evolution function.

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            The result of the optimisation including the optimised parameter values and cost.
        """
        self._iterations = 0

        # Add callback storing history of parameter values
        def callback(intermediate_result: OptimizeResult):
            (x_search, x_model, cost) = self.get_and_reset_itermediate_results()
            self._iterations += 1
            self.logger.log_update(
                iterations=self._iterations,
                evaluations=len(x_search),
                x_search_best=intermediate_result.x,
                x_model_best=self.problem.params.transformation.to_model(
                    intermediate_result.x
                ),
                cost_best=intermediate_result.fun,
                x_search=x_search,
                x_model=x_model,
                cost=cost,
            )

        def cost_wrapper(x):
            x_model = self.problem.params.transformation.to_model(x)
            cost = self._evaluator.evaluate(x_model)
            self.store_intermediate_results(x_search=x, x_model=x_model, cost=cost)
            return cost

        start_time = time()
        result = differential_evolution(
            cost_wrapper,
            self._scipy_bounds,
            callback=callback,
            **self._options_dict,
        )
        total_time = time() - start_time

        try:
            nit = result.nit
        except AttributeError:
            nit = -1
        try:
            nfev = result.nfev
        except AttributeError:
            nfev = -1

        return OptimisationResult(
            best_cost=result.fun,
            n_evaluations=nfev,
            problem=self.problem,
            x=result.x,
            n_iterations=nit,
            time=total_time,
            message=result.message,
        )

    def name(self):
        """Provides the name of the optimisation strategy."""
        return "SciPyDifferentialEvolution"
