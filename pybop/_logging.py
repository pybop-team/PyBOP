import numpy as np


class Logger:
    """
    Records the parameter values and corresponding cost values.

    Parameters
    ----------
    verbose : bool
        If True, the optimisation progress and results are printed.
    verbose_print_rate : int
        The distance between iterations to print verbose output.
    iteration : int
        The current iteration number.
    x_model : list[np.ndarray]
        The history of model parameters.
    x_search : list[np.ndarray]
        The history of search parameters.
    cost : list[float]
        The history of the cost value.
    iteration_number : list[int]
        The history of the iteration number.
    evaluations : int
        The current number of evaluations.
    x_model_best : list[np.ndarray]
        The current best model parameters.
    cost_best : list[float]
        The current best cost value.
    """

    def __init__(
        self, minimising: bool, verbose: bool = False, verbose_print_rate: int = 50
    ):
        self._minimising = minimising
        self.verbose = verbose
        self.verbose_print_rate = verbose_print_rate
        self.iteration = None
        self.reset()

    def reset(self):
        self.x_model = []
        self.x_search = []
        self.cost = []
        self.iteration_number = []
        self.evaluations = 0
        self.x_model_best = []
        self.cost_best = None

    @property
    def x0(self):
        """Get the initial parameter values."""
        return self.x_model[0] if self.x_model else None

    @property
    def cost_convergence(self):
        """Get the convergence of the cost during the optimisation."""
        if self._minimising:
            return np.minimum.accumulate(self.cost)
        return np.maximum.accumulate(self.cost)

    def extend_log(
        self, x_model: list[np.ndarray], x_search: list[np.ndarray], cost: list[float]
    ):
        """
        Update the log with new values.

        Parameters
        ----------
        x_model : list[np.ndarray]
            The model parameters.
        x_search : list[np.ndarray]
            The search parameters.
        cost : list[float]
            The cost associated with the parameters.
        """
        # Update logs for each provided parameter
        self.x_model.extend(x_model)
        self.x_search.extend(x_search)
        self.cost.extend(cost)

        # Update counts using iteration number from optimiser
        evals = len(cost)
        self.iteration_number.extend([self.iteration] * evals)
        self.evaluations += evals

        # Update best values
        if self._minimising:
            i = np.nanargmin(self.cost)
        else:
            i = np.nanargmax(self.cost)
        self.x_model_best = self.x_model[i]
        self.x_search_best = self.x_search[i]
        self.cost_best = self.cost[i]

        # Verbose output
        self._print_verbose_output()

    def _print_verbose_output(self):
        """Print verbose optimisation information if enabled."""
        if not self.verbose:
            return

        # Only print on first 10 evaluations, then every Nth iteration
        if self.iteration > 10 and self.iteration % self.verbose_print_rate != 0:
            return

        print(
            f"| Iter: {self.iteration} | Evals: {self.evaluations}"
            f"| Best Parameters: {self.x_model_best} | Best Cost: {self.cost_best}"
        )
