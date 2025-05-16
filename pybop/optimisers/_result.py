from typing import Optional

import numpy as np
import scipy.optimize

from pybop.problems.base_problem import Problem


class OptimisationResult:
    """
    Stores the result of the optimisation.

    Attributes
    ----------
    problem: pybop.Problem
        The optimisation object used to generate the results.
    x : ndarray
        The solution of the optimisation (in model space)
    final_cost : float
        The cost associated with the solution x.
    n_iterations : int
        Number of iterations performed by the optimiser.
    message : str
        The reason for stopping given by the optimiser.
    scipy_result : scipy.optimize.OptimizeResult, optional
        The result obtained from a SciPy optimiser.
    """

    def __init__(
        self,
        problem: Problem,
        x: np.ndarray,
        final_cost: float,
        n_iterations: int,
        n_evaluations: int,
        time: float,
        sensitivities: Optional[dict] = None,
        message: Optional[str] = None,
        scipy_result: Optional[scipy.optimize.OptimizeResult] = None,
    ):
        self._problem = problem
        self.n_runs = 0
        self._best_run = None
        self._x = [x]
        self._final_cost = [final_cost]
        self._sensitivities = sensitivities if sensitivities is not None else {}
        self._n_iterations = [n_iterations]
        self._n_evaluations = [n_evaluations]
        self._message = [message]
        self._scipy_result = [scipy_result]
        self._time = [time]

        x0 = self._problem.params.initial_value()
        self._x0 = [x0]

        # Calculate Fisher Information if available
        try:
            fisher = self._problem.observed_fisher(x)
            diag_fish = np.diag(fisher) if fisher is not None else None
        except NotImplementedError:
            diag_fish = None
        self._fisher = [diag_fish]

        self._validate()

    @staticmethod
    def combine(results: list["OptimisationResult"]) -> "OptimisationResult":
        """
        Combine multiple OptimisationResult objects into a single one.

        Parameters
        ----------
        results : list[OptimisationResult]
            List of OptimisationResult objects to combine.

        Returns
        -------
        OptimisationResult
            Combined OptimisationResult object.
        """
        if len(results) == 0:
            raise ValueError("No results to combine.")
        ret = results[0]
        ret._x = [x for result in results for x in result._x]
        ret._final_cost = [x for result in results for x in result._final_cost]
        ret._fisher = [x for result in results for x in result._fisher]
        ret._n_iterations = [x for result in results for x in result._n_iterations]
        ret._n_evaluations = [x for result in results for x in result._n_evaluations]
        ret._message = [x for result in results for x in result._message]
        ret._scipy_result = [x for result in results for x in result._scipy_result]
        ret._time = [x for result in results for x in result._time]
        ret._x0 = [x for result in results for x in result._x0]
        ret._sensitivities = {}
        for result in results:
            for key, value in result._sensitivities.items():
                if key not in ret._sensitivities:
                    ret._sensitivities[key] = []
                ret._sensitivities[key].extend(value)
        ret._best_run = None
        ret.n_runs = len(results)
        ret._validate()
        return ret

    def _validate(self):
        # Check that there is a finite cost and update best run
        self._check_for_finite_cost()
        self._best_run = self._final_cost.index(min(self._final_cost))

    def _check_for_finite_cost(self) -> None:
        """
        Validate the optimised parameters and ensure they produce a finite cost value.

        Raises:
            ValueError: If the optimised parameters do not produce a finite cost value.
        """
        if not any(np.isfinite(self._final_cost)):
            raise ValueError(
                f"Optimised parameters {self._problem.params.as_dict()} do not produce a finite cost value"
            )

    def __str__(self) -> str:
        """
        A string representation of the OptimisationResult object.

        Returns:
            str: A formatted string containing optimisation result information.
        """
        # Format the sensitivities
        self.sense_format = ""
        if self._sensitivities:
            for value, conf in zip(
                self._sensitivities["ST"], self._sensitivities["ST_conf"]
            ):
                self.sense_format += f" {value:.3f} ± {conf:.3f},"

        return (
            f"OptimisationResult:\n"
            f"  Best result from {self.n_runs} run(s).\n"
            f"  Initial parameters: {self.x0_best}\n"
            f"  Optimised parameters: {self.x_best}\n"
            f"  Total-order sensitivities:{self.sense_format}\n"
            f"  Diagonal Fisher Information entries: {self.fisher_best}\n"
            f"  Final cost: {self.final_cost_best}\n"
            f"  Optimisation time: {self.time_best} seconds\n"
            f"  Number of iterations: {self.n_iterations_best}\n"
            f"  Number of evaluations: {self.n_evaluations_best}\n"
            f"  Reason for stopping: {self.message_best}\n"
            f"  SciPy result available: {'Yes' if self.scipy_result_best else 'No'}\n"
        )

    def average_iterations(self) -> Optional[np.floating]:
        """Calculates the average number of iterations across all runs."""
        return np.mean(self._n_iterations) if len(self._n_iterations) > 0 else None

    def total_runtime(self) -> Optional[np.floating]:
        """Calculates the total runtime across all runs."""
        return np.sum(self._time) if len(self._time) > 0 else None

    def _get_single_or_all(self, attr):
        value = getattr(self, attr)
        return value[0] if len(value) == 1 else value

    @property
    def x(self):
        return self._get_single_or_all("_x")

    @property
    def x_best(self):
        return self._x[self._best_run] if self._best_run is not None else None

    @property
    def x0(self):
        return self._get_single_or_all("_x0")

    @property
    def x0_best(self):
        return self._x0[self._best_run] if self._best_run is not None else None

    @property
    def final_cost(self):
        return self._get_single_or_all("_final_cost")

    @property
    def final_cost_best(self):
        return self._final_cost[self._best_run] if self._best_run is not None else None

    @property
    def fisher(self):
        return self._get_single_or_all("_fisher")

    @property
    def sensitivities(self):
        return self._get_single_or_all("_sensitivities")

    @sensitivities.setter
    def sensitivities(self, obj: dict):
        self._sensitivities = obj

    @property
    def fisher_best(self):
        return self._fisher[self._best_run] if self._best_run is not None else None

    @property
    def n_iterations(self):
        return self._get_single_or_all("_n_iterations")

    @property
    def n_iterations_best(self):
        return (
            self._n_iterations[self._best_run] if self._best_run is not None else None
        )

    @property
    def n_evaluations(self):
        return self._get_single_or_all("_n_evaluations")

    @property
    def n_evaluations_best(self):
        return (
            self._n_evaluations[self._best_run] if self._best_run is not None else None
        )

    @property
    def message(self):
        return self._get_single_or_all("_message")

    @property
    def message_best(self):
        return self._message[self._best_run] if self._best_run is not None else None

    @property
    def scipy_result(self):
        return self._get_single_or_all("_scipy_result")

    @property
    def scipy_result_best(self):
        return (
            self._scipy_result[self._best_run] if self._best_run is not None else None
        )

    @property
    def time(self):
        return self._get_single_or_all("_time")

    @property
    def time_best(self):
        return self._time[self._best_run] if self._best_run is not None else None
