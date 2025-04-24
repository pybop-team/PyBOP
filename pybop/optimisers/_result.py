import warnings
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from pybamm import Solution

from pybop import BaseCost, BaseLikelihood, Inputs

if TYPE_CHECKING:
    from pybop import BaseOptimiser


class OptimisationResult:
    """
    Stores the result of the optimisation.

    Attributes
    ----------
    optim : pybop.BaseOptimiser
        The optimisation object used to generate the results.
    x : ndarray
        The solution of the optimisation.
    final_cost : float
        The cost associated with the solution x.
    n_iterations : int
        Number of iterations performed by the optimiser.
    message : str
        The reason for stopping given by the optimiser.
    scipy_result : scipy.optimize.OptimizeResult, optional
        The result obtained from a SciPy optimiser.
    pybamm_solution: pybamm.Solution or list[pybamm.Solution], optional
        The best solution object(s) obtained from the optimisation.
    """

    def __init__(
        self,
        optim: "BaseOptimiser",
        x: Union[Inputs, np.ndarray] = None,
        final_cost: Optional[float] = None,
        sensitivities: Optional[dict] = None,
        n_iterations: Optional[int] = None,
        n_evaluations: Optional[int] = None,
        time: Optional[float] = None,
        message: Optional[str] = None,
        scipy_result=None,
    ):
        self.optim = optim
        self.cost = self.optim.cost
        self.minimising = not self.optim.invert_cost
        self._transformation = self.optim.transformation
        self.n_runs = 0
        self._best_run = None
        self._x = []
        self._final_cost = []
        self._sensitivities = None
        self._fisher = []
        self._n_iterations = []
        self._n_evaluations = []
        self._message = []
        self._scipy_result = []
        self._time = []
        self._x0 = []
        self._pybamm_solution = []

        if x is not None:
            # Transform the parameter values and update the sign of any final cost
            # coming directly from an optimiser
            x = self._transformation.to_model(x) if self._transformation else x
            final_cost = (
                final_cost * (1 if self.minimising else -1)
                if final_cost is not None
                else self.cost(x)
            )
            x0 = self.optim.parameters.initial_value()

            # Evaluate the problem once more to update the solution
            try:
                self.cost(x)
                pybamm_solution = self.cost.pybamm_solution
            except Exception:
                warnings.warn(
                    "Failed to evaluate the model with best fit parameters.",
                    UserWarning,
                    stacklevel=2,
                )
                pybamm_solution = None

            # Calculate Fisher Information if Likelihood
            if isinstance(self.cost, BaseLikelihood):
                fisher = self.cost.observed_fisher(x)
                diag_fish = np.diag(fisher) if fisher is not None else None
            else:
                diag_fish = None

            self._extend(
                x=[x],
                final_cost=[final_cost],
                fisher=[diag_fish],
                n_iterations=[n_iterations],
                n_evaluations=[n_evaluations],
                time=[time],
                message=[message],
                scipy_result=[scipy_result],
                x0=[x0],
                pybamm_solution=[pybamm_solution],
            )

    def add_result(self, result):
        """Add a preprocessed OptimisationResult."""
        self._extend(
            x=result._x,  # noqa: SLF001
            final_cost=result._final_cost,  # noqa: SLF001
            fisher=result._fisher,  # noqa: SLF001
            n_iterations=result._n_iterations,  # noqa: SLF001
            n_evaluations=result._n_evaluations,  # noqa: SLF001
            time=result._time,  # noqa: SLF001
            message=result._message,  # noqa: SLF001
            scipy_result=result._scipy_result,  # noqa: SLF001
            x0=result._x0,  # noqa: SLF001
            pybamm_solution=result._pybamm_solution,  # noqa: SLF001
        )

    def _extend(
        self,
        x: Union[list[Inputs], list[np.ndarray]],
        final_cost: list[float],
        fisher: list,
        n_iterations: list[int],
        n_evaluations: list[int],
        time: list[float],
        message: list[str],
        scipy_result: list,
        x0: list,
        pybamm_solution: list[Solution],
    ):
        self.n_runs += len(final_cost)
        self._x.extend(x)
        self._final_cost.extend(final_cost)
        self._fisher.extend(fisher)
        self._n_iterations.extend(n_iterations)
        self._n_evaluations.extend(n_evaluations)
        self._message.extend(message)
        self._scipy_result.extend(scipy_result)
        self._time.extend(time)
        self._x0.extend(x0)
        self._pybamm_solution.extend(pybamm_solution)

        # Check that there is a finite cost and update best run
        self.check_for_finite_cost()
        self._best_run = self._final_cost.index(
            min(self._final_cost) if self.minimising else max(self._final_cost)
        )

        # Check that the best parameters are physically viable
        self.check_physical_viability(self.x_best)

    def check_for_finite_cost(self) -> None:
        """
        Validate the optimised parameters and ensure they produce a finite cost value.

        Raises:
            ValueError: If the optimised parameters do not produce a finite cost value.
        """
        if not any(np.isfinite(self._final_cost)):
            raise ValueError(
                f"Optimised parameters {self.cost.parameters.as_dict()} do not produce a finite cost value"
            )

    def check_physical_viability(self, x):
        """
        Check if the optimised parameters are physically viable.

        Parameters
        ----------
        x : array-like
            Optimised parameter values.
        """
        if (
            not isinstance(self.cost, BaseCost)
            or self.cost.problem is None
            or self.cost.problem.model is None
        ):
            warnings.warn(
                "No model within problem class, can't check physical viability.",
                UserWarning,
                stacklevel=2,
            )
            return

        if self.cost.problem.model.check_params(
            inputs=x, allow_infeasible_solutions=False
        ):
            return
        else:
            warnings.warn(
                "Optimised parameters are not physically viable! \nConsider retrying the optimisation"
                " with a non-gradient-based optimiser and the option allow_infeasible_solutions=False",
                UserWarning,
                stacklevel=2,
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
            f"  PyBaMM Solution available: {'Yes' if self.pybamm_solution else 'No'}"
        )

    def average_iterations(self) -> Optional[float]:
        """Calculates the average number of iterations across all runs."""
        return np.mean(self._n_iterations)

    def total_runtime(self) -> Optional[float]:
        """Calculates the total runtime across all runs."""
        return np.sum(self._time)

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
    def pybamm_solution(self):
        return self._get_single_or_all("_pybamm_solution")

    @property
    def pybamm_solution_best(self):
        return (
            self._pybamm_solution[self._best_run]
            if self._best_run is not None
            else None
        )

    @property
    def time(self):
        return self._get_single_or_all("_time")

    @property
    def time_best(self):
        return self._time[self._best_run] if self._best_run is not None else None
