import warnings
from copy import deepcopy
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
import pybop


@dataclass
class OptimisationLog:
    """Stores optimisation progress data."""

    iterations: list[int] = field(default_factory=list)
    evaluations: list[int] = field(default_factory=list)
    x: list[list[float]] = field(default_factory=list)
    x_best: list[list[float]] = field(default_factory=list)
    x_search: list[list[float]] = field(default_factory=list)
    x0: list[list[float]] = field(default_factory=list)
    cost: list[float] = field(default_factory=list)
    cost_best: list[float] = field(default_factory=list)


class OptimisationLogger:
    def __init__(self, verbose: bool = False, verbose_print_rate: int = 50):
        self._verbose = verbose
        self._verbose_print_rate = verbose_print_rate
        self._iter_count = 0

    def log_update(
        self,
        iterations=None,
        evaluations=None,
        x_model=None,
        x_search=None,
        x_model_best=None,
        x_seach_best=None,
        cost=None,
        cost_best=None,
        x0=None,
    ):
        """
        Update the log with new values.

        Parameters
        ----------
        iterations : list or array-like, optional
            Iteration indices to log (default: None).
        evaluations: list or array-like, optional
            Evaluation indices to log (default: None).
        x_model : list or array-like, optional
            Parameter values in model space (default: None).
        x_search : list or array-like, optional
            Search parameter values in search space (default: None).
        x_best_model : list or array-like, optional
            Parameter values in model space corresponding to the best cost yet (default: None).
        x_best_search : list or array-like, optional
            Parameter values in search space corresponding to the best cost yet (default: None).
        cost : list, optional
            Cost values corresponding to x (default: None).
        cost_best : list, optional
            Cost values corresponding to x_best (default: None).
        x0 : list or array-like, optional
            Initial parameter values (default: None).
        """
        # Update logs for each provided parameter
        self._update_log_entry("iterations", iterations)
        self._update_log_entry("evaluations", evaluations)
        self._update_log_entry("x_model", x_model)
        self._update_log_entry("x_search", x_search)
        self._update_log_entry("x_model_best", x_model_best)
        self._update_log_entry("x_search_best", x_seach_best)
        self._update_log_entry("x0", x0)
        self._update_log_entry("cost", cost)
        self._update_log_entry("cost_best", cost_best)

        # Verbose output
        self._print_verbose_output()
        self._iter_count += 1

    def _update_log_entry(self, key, value):
        """Update a log entry if the value is provided."""
        if value is not None:
            getattr(self.log, key).extend(self._to_list(value))

    def _to_list(self, array_like):
        """Convert input to a list."""
        if isinstance(array_like, (list, tuple, np.ndarray, jnp.ndarray)):
            return list(array_like)
        return [array_like]

    def _print_verbose_output(self):
        """Print verbose optimization information if enabled."""
        if not self._verbose:
            return

        latest_iter = (
            self.log.iterations[-1] if self.log.iterations else self._iter_count
        )

        # Only print on first 10 iterations, then every Nth iteration
        if latest_iter > 10 and latest_iter % self._verbose_print_rate != 0:
            return

        latest_eval = self.log.evaluations[-1] if self.log.evaluations else "N/A"
        latest_x_best = self.log.x_best[-1] if self.log.x_best else "N/A"
        latest_cost_best = self.log.cost_best[-1] if self.log.cost_best else "N/A"

        print(
            f"Iter: {latest_iter} | Evals: {latest_eval} | "
            f"Best Values: {latest_x_best} | Best Cost: {latest_cost_best} |"
        )


class OptimiserImpl:
    def _set_up_optimiser(self):
        """
        Parse optimiser options and prepare the optimiser.

        This method should be implemented by child classes.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the subclass.
        """
        raise NotImplementedError

    def _run(self):
        """
        Contains the logic for the optimisation algorithm.

        This method should be implemented by child classes to perform the actual optimisation.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the subclass.
        """
        raise NotImplementedError

    def name(self):
        """
        Returns the name of the optimiser, to be overwritten by child classes.

        Returns
        -------
        str
            The name of the optimiser
        """
        raise NotImplementedError  # pragma: no cover


class Optimiser:
    """
    A base class for defining optimisation methods.

    This class serves as a base class for creating optimisers. It provides a basic structure for
    an optimisation algorithm, including the initial setup and a method stub for performing the
    optimisation process. Child classes should override _set_up_optimiser and the _run method with
    a specific algorithm.

    Parameters
    ----------
    problem : pybop.Problem
        An objective function to be optimised.
    x0 : np.ndarray
        Initial values of the parameters for the optimisation.
    sigma0 : np.ndarray (optional)
        Initial step size or standard deviation in the (search) parameters. Either a scalar value
        (same for all coordinates) or an array with one entry per dimension.
        Not all methods will use this information.
    logger: pybop.OptimisationLogger (optional)
        An object to log the optimisation progress. If None, the default logger is used.
    physical_viability : bool, optional
        If True, the feasibility of the optimised parameters is checked (default: False).
    allow_infeasible_solutions : bool, optional
        If True, infeasible parameter values will be allowed in the optimisation (default: True).

    """

    def __init__(
        self,
        problem: pybop.Problem,
        logger: OptimisationLogger | None = None,
        physical_viability: bool = False,
        allow_infeasible_solutions: bool = True,
    ):

        self._problem = problem
        self._logger = logger or OptimisationLogger()
        self._physical_viability = physical_viability
        self._allow_infeasible_solutions = allow_infeasible_solutions
        self._set_up_optimiser()

    def run(self):
        """
        Run the optimisation and return the optimised parameters and final cost.

        Returns
        -------
        results: OptimisationResult
            The pybop optimisation result class.
        """
        self.result = pybop.OptimisationResult(optim=self)

        for i in range(self.multistart):
            if i >= 1:
                self.unset_options = self.unset_options_store.copy()
                self.parameters.update(initial_values=self.parameters.rvs(1))
                self.x0 = self.parameters.reset_initial_value(apply_transform=True)
                self._set_up_optimiser()

            self.result.add_result(self._run())

        # Store the optimised parameters
        self.parameters.update(values=self.result.x_best)

        # Compute sensitivities
        self.result.sensitivities = self._parameter_sensitivities()

        if self.verbose:
            print(self.result)

        return self.result

    def set_allow_infeasible_solutions(self, allow: bool = True):
        """
        Set whether to allow infeasible solutions or not.

        Parameters
        ----------
        iterations : bool, optional
            Whether to allow infeasible solutions.
        """
        # Set whether to allow infeasible locations
        self.physical_viability = allow
        self.allow_infeasible_solutions = allow
