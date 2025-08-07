import numpy as np
from pybamm import ParameterValues

from pybop import Problem, PybammEISProblem, PybammProblem


class OptimisationResult:
    """
    Stores the result of the optimisation.

    Attributes
    ----------
    problem: pybop.Problem
        The optimisation object used to generate the results.
    x : ndarray
        The solution of the optimisation (in model space)
    best_cost : float
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
        best_cost: float,
        n_iterations: int,
        n_evaluations: int,
        time: float,
        message: str | None = None,
    ):
        self._problem = problem
        self.n_runs = 0
        self._best_run = None
        self._x = [x]
        self._best_cost = [best_cost]
        self._n_iterations = [n_iterations]
        self._n_evaluations = [n_evaluations]
        self._message = [message]
        self._time = [time]
        self._parameter_values = self._set_optimal_parameter_values()

        x0 = self._problem.params.get_initial_values()
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
        ret._x = [x for result in results for x in result._x]  # noqa: SLF001
        ret._parameter_values = [result._parameter_values for result in results]  # noqa: SLF001
        ret._best_cost = [  # noqa: SLF001
            x
            for result in results
            for x in result._best_cost  # noqa: SLF001
        ]
        ret._fisher = [x for result in results for x in result._fisher]  # noqa: SLF001
        ret._n_iterations = [  # noqa: SLF001
            x
            for result in results
            for x in result._n_iterations  # noqa: SLF001
        ]
        ret._n_evaluations = [  # noqa: SLF001
            x
            for result in results
            for x in result._n_evaluations  # noqa: SLF001
        ]
        ret._message = [  # noqa: SLF001
            x
            for result in results
            for x in result._message  # noqa: SLF001
        ]
        ret._time = [x for result in results for x in result._time]  # noqa: SLF001
        ret._x0 = [x for result in results for x in result._x0]  # noqa: SLF001
        ret._best_run = None  # noqa: SLF001
        ret.n_runs = len(results)
        ret._validate()  #  noqa: SLF001

        return ret

    def _set_optimal_parameter_values(self) -> ParameterValues | dict:
        if isinstance(self._problem, PybammProblem | PybammEISProblem):
            pybamm_params = self._problem.pipeline.parameter_values
            for i, param in enumerate(self._problem.params):
                pybamm_params.update({param.name: self._x[0][i]})
            return pybamm_params

        return {}

    def _validate(self):
        # Check that there is a finite cost and update best run
        self._check_for_finite_cost()
        self._best_run = self._best_cost.index(min(self._best_cost))

    def _check_for_finite_cost(self) -> None:
        """
        Validate the optimised parameters and ensure they produce a finite cost value.

        Raises:
            ValueError: If the optimised parameters do not produce a finite cost value.
        """
        if not any(np.isfinite(self._best_cost)):
            raise ValueError(
                f"Optimised parameters {self._problem.params.to_dict()} do not produce a finite cost value"
            )

    def __str__(self) -> str:
        """
        A string representation of the OptimisationResult object.

        Returns:
            str: A formatted string containing optimisation result information.
        """
        return (
            f"OptimisationResult:\n"
            f"  Best result from {self.n_runs} run(s).\n"
            f"  Initial parameters: {self.x0}\n"
            f"  Optimised parameters: {self.x}\n"
            f"  Diagonal Fisher Information entries: {self.fisher}\n"
            f"  Best cost: {self.best_cost}\n"
            f"  Optimisation time: {self.time} seconds\n"
            f"  Number of iterations: {self.total_iterations()}\n"
            f"  Number of evaluations: {self.total_evaluations()}\n"
            f"  Reason for stopping: {self.message}"
        )

    def total_iterations(self) -> np.floating | None:
        """Calculates the average number of iterations across all runs."""
        return np.sum(self._n_iterations) if len(self._n_iterations) > 0 else None

    def total_evaluations(self) -> np.floating | None:
        """Calculates the average number of iterations across all runs."""
        return np.sum(self._n_iterations) if len(self._n_iterations) > 0 else None

    def total_runtime(self) -> np.floating | None:
        """Calculates the total runtime across all runs."""
        return np.sum(self._time) if len(self._time) > 0 else None

    def _get_single_or_all(self, attr):
        value = getattr(self, attr)
        if len(value) > 1:
            return value[self._best_run]
        return value[0]

    @property
    def x(self) -> np.ndarray:
        """The solution of the optimisation (in model space)."""
        return self._get_single_or_all("_x")

    @property
    def parameter_values(self) -> ParameterValues | dict:
        """The parameter values from the optimisation."""
        return self._get_single_or_all("_parameter_values")

    @property
    def x0(self) -> np.ndarray:
        """The initial parameter values."""
        return self._get_single_or_all("_x0")

    @property
    def best_cost(self) -> float:
        """The final cost value(s)."""
        return self._get_single_or_all("_best_cost")

    @property
    def fisher(self) -> np.ndarray | None:
        """The Fisher information matrix diagonal."""
        return self._get_single_or_all("_fisher")

    @property
    def n_iterations(self) -> int:
        """The number of iterations."""
        return self._get_single_or_all("_n_iterations")

    @property
    def n_evaluations(self) -> int:
        """The number of evaluations."""
        return self._get_single_or_all("_n_evaluations")

    @property
    def message(self) -> str | None:
        """The optimization termination message(s)."""
        return self._get_single_or_all("_message")

    @property
    def time(self) -> float | None:
        """The optimization time(s)."""
        return self.total_runtime()
