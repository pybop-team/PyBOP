import numpy as np
from pybamm import ParameterValues

from pybop import Logger, Problem, PybammEISProblem, PybammProblem, plot


class OptimisationResult:
    """
    Stores the result of the optimisation.

    Attributes
    ----------
    problem : pybop.Problem
        The optimisation problem.
    logger : pybop.Logger
        The log of the optimisation process.
    time : float
        The time taken.
    optim_name : str
        The name of the optimiser.
    message : str
        The reason for stopping given by the optimiser.
    scipy_result : scipy.optimize.OptimizeResult, optional
        The result obtained from a SciPy optimiser.
    """

    def __init__(
        self,
        problem: Problem,
        logger: Logger,
        time: float,
        optim_name: str | None = None,
        message: str | None = None,
        scipy_result=None,
    ):
        self._problem = problem
        self.optim_name = optim_name
        self.n_runs = 0
        self._best_run = None
        self._parameter_values = None
        self._x = [logger.x_model_best]
        self._x_model = [logger.x_model]
        self._x0 = [logger.x0]
        self._best_cost = [logger.cost_best]
        self._cost = [logger.cost_convergence]
        self._initial_cost = [logger.cost[0]]
        self._n_iterations = [logger.iteration]
        self._iteration_number = [logger.iteration_number]
        self._n_evaluations = [logger.evaluations]
        self._message = [message]
        self._time = [time]

        # Calculate Fisher Information if available
        try:
            fisher = self._problem.observed_fisher(logger.x_model_best)
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
        ret._x_model = [x for result in results for x in result._x_model]  # noqa: SLF001
        ret._x0 = [x for result in results for x in result._x0]  # noqa: SLF001
        ret._best_cost = [  # noqa: SLF001
            x
            for result in results
            for x in result._best_cost  # noqa: SLF001
        ]
        ret._cost = [x for result in results for x in result._cost]  # noqa: SLF001
        ret._initial_cost = [  # noqa: SLF001
            x
            for result in results
            for x in result._initial_cost  # noqa: SLF001
        ]
        ret._fisher = [x for result in results for x in result._fisher]  # noqa: SLF001
        ret._n_iterations = [  # noqa: SLF001
            x
            for result in results
            for x in result._n_iterations  # noqa: SLF001
        ]
        ret._iteration_number = [  # noqa: SLF001
            x
            for result in results
            for x in result._iteration_number  # noqa: SLF001
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

        ret._best_run = None  # noqa: SLF001
        ret.n_runs = len(results)
        ret._validate()  #  noqa: SLF001

        return ret

    def _validate(self):
        """Check that there is a finite cost and update best run."""
        self._check_for_finite_cost()
        self._best_run = self._best_cost.index(min(self._best_cost))
        self._parameter_values = self._set_optimal_parameter_values()

    def _set_optimal_parameter_values(self) -> ParameterValues | dict:
        if isinstance(self._problem, PybammProblem | PybammEISProblem):
            pybamm_params = self._problem.pipeline.parameter_values
            for i, param in enumerate(self._problem.params):
                pybamm_params.update({param.name: self._x[0][i]})
            return pybamm_params

        return {}

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
        """Calculates the average number of evaluations across all runs."""
        return np.sum(self._n_evaluations) if len(self._n_evaluations) > 0 else None

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
    def x_model(self) -> np.ndarray:
        """The log of the evaluated parameters (in model space)."""
        return self._get_single_or_all("_x_model")

    @property
    def x0(self) -> np.ndarray:
        """The initial parameter values."""
        return self._get_single_or_all("_x0")

    @property
    def best_cost(self) -> float:
        """The best cost value(s)."""
        return self._get_single_or_all("_best_cost")

    @property
    def cost(self) -> np.ndarray:
        """The log of the cost values."""
        return self._get_single_or_all("_cost")

    @property
    def initial_cost(self) -> float:
        """The initial cost value(s)."""
        return self._get_single_or_all("_initial_cost")

    @property
    def fisher(self) -> np.ndarray | None:
        """The Fisher information matrix diagonal."""
        return self._get_single_or_all("_fisher")

    @property
    def n_iterations(self) -> int:
        """The number of iterations."""
        return self._get_single_or_all("_n_iterations")

    @property
    def iteration_number(self) -> np.ndarray | None:
        """The number of iterations."""
        return self._get_single_or_all("_iteration_number")

    @property
    def n_evaluations(self) -> int:
        """The number of evaluations."""
        return self._get_single_or_all("_n_evaluations")

    @property
    def problem(self) -> Problem:
        """The optimisation problem."""
        return self._problem

    @property
    def parameter_values(self) -> ParameterValues | dict:
        """The best parameter values from the optimisation."""
        return self._parameter_values

    @property
    def message(self) -> str | None:
        """The optimisation termination message(s)."""
        return self._get_single_or_all("_message")

    @property
    def time(self) -> float | None:
        """The optimisation time(s)."""
        return self.total_runtime()

    def plot_convergence(self, **kwargs):
        """
        Plot the evolution of the best cost during the optimisation.

        Parameters
        ----------
        show : bool, optional
            If True, the figure is shown upon creation (default: True).
        **layout_kwargs : optional
            Valid Plotly layout keys and their values.
        """
        return plot.convergence(self, **kwargs)

    def plot_parameters(self, **kwargs):
        """
        Plot the evolution of parameter values during the optimisation.

        Parameters
        ----------
        show : bool, optional
            If True, the figure is shown upon creation (default: True).
        **layout_kwargs : optional
            Valid Plotly layout keys and their values.
        """
        return plot.parameters(self, **kwargs)

    def plot_surface(self, **kwargs):
        """
        Plot a 2D representation of the Voronoi diagram with color-coded regions.

        Parameters
        ----------
        bounds : numpy.ndarray, optional
            A 2x2 array specifying the [min, max] bounds for each parameter.
        normalise : bool, optional
            If True, the voronoi regions are computed using the Euclidean distance between
            points normalised with respect to the bounds (default: True).
        resolution : int, optional
            Resolution of the plot (default: 500).
        show : bool, optional
            If True, the figure is shown upon creation (default: True).
        **layout_kwargs : optional
            Valid Plotly layout keys and their values.
        """
        return plot.surface(self, **kwargs)

    def plot_contour(self, **kwargs):
        """
        Generate and plot a 2D visualisation of the cost landscape with the optimisation trace.

        Parameters
        ----------
        gradient : bool, optional
            If True, gradient plots are also generated (default: False).
        bounds : np.ndarray, optional
            A 2x2 array specifying the [min, max] bounds for each parameter.
        apply_transform : bool, optional
            Uses the transformed parameter values, as seen by the optimiser (default: False).
        steps : int, optional
            The number of grid points to divide the parameter space into along each dimension
            (default: 10).
        show : bool, optional
            If True, the figure is shown upon creation (default: True).
        use_optim_log : bool, optional
            If True, the optimisation log is used to inform the cost landscape (default: False).
        **layout_kwargs : optional
            Valid Plotly layout keys and their values.
        """
        return plot.contour(self, **kwargs)
