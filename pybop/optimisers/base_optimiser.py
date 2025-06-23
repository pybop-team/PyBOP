from dataclasses import dataclass

import numpy as np

import pybop
from pybop.problems.base_problem import Problem


class OptimisationLogger:
    def __init__(self, verbose: bool = False, verbose_print_rate: int = 50):
        self._verbose = verbose
        self._verbose_print_rate = verbose_print_rate
        self.x_model = []
        self.x_search = []
        self.x_model_best = []
        self.x_search_best = []
        self.cost = []
        self.cost_best = []
        self.iterations = []
        self.evaluations = []

    @property
    def verbose(self):
        """Get the verbosity level."""
        return self._verbose

    @property
    def x0(self):
        """Get the initial parameter values."""
        if self.x_model:
            return self.x_model[0]
        return None

    @property
    def last_x_model_best(self):
        """Get the best model parameters found during optimisation."""
        if self.x_model_best:
            return self.x_model_best[-1]
        return None

    def log_update(
        self,
        x_model: list[np.ndarray],
        x_search: list[np.ndarray],
        cost: list[float],
        iterations: int,
        evaluations: int,
        x_model_best: np.ndarray,
        x_search_best: np.ndarray,
        cost_best: float,
    ):
        """
        Update the log with new values.

        Parameters
        ----------
        x_model : list[float]
            The model parameters.
        x_search : list[float]
            The search parameters.
        cost : list[float]
            The cost associated with the parameters.
        iterations : int, optional
            The number of iterations performed, by default None
        evaluations : int, optional
            The number of evaluations performed, by default None
        x_model_best : float, optional
            The best model parameters found, by default None
        x_search_best : float, optional
            The best search parameters found, by default None
        cost_best : float, optional
            The best cost associated with the parameters, by default None

        """
        # Update logs for each provided parameter
        self.x_model.extend(x_model)
        self.x_search.extend(x_search)
        self.cost.extend(cost)
        self.iterations.append(iterations)
        self.evaluations.append(evaluations)
        self.x_model_best.append(x_model_best)
        self.x_search_best.append(x_search_best)
        self.cost_best.append(cost_best)

        # Verbose output
        self._print_verbose_output()

    def _print_verbose_output(self):
        """Print verbose optimisation information if enabled."""
        if not self._verbose:
            return

        latest_iter = self.iterations[-1]

        # Only print on first 10 iterations, then every Nth iteration
        if latest_iter > 10 and latest_iter % self._verbose_print_rate != 0:
            return

        latest_eval = self.evaluations[-1]
        latest_x_best = self.x_model_best[-1]
        latest_cost_best = self.cost_best[-1]

        print(
            f"Iter: {latest_iter} | Evals: {latest_eval} | "
            f"Best Values: {latest_x_best} | Best Cost: {latest_cost_best} |"
        )


@dataclass
class OptimiserOptions:
    multistart: int = 1
    verbose: bool = False
    verbose_print_rate: int = 50

    def validate(self):
        """
        Validate the options.

        Raises
        ------
        ValueError
            If the options are invalid.
        """
        if self.multistart < 1:
            raise ValueError("Multistart must be greater than or equal to 1.")
        if self.verbose_print_rate < 1:
            raise ValueError("Verbose print rate must be greater than or equal to 1.")


class BaseOptimiser:
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
    logger: pybop.OptimisationLogger
        An object to log the optimisation progress.
    options: pybop.OptimiserOptions (optional)
        Options for the optimiser, such as multistart.


    """

    default_max_iterations = 1000

    def __init__(
        self,
        problem: Problem,
        options: OptimiserOptions | None = None,
    ):
        if not isinstance(problem, Problem):
            raise TypeError(f"Expected a pybop.Problem instance, got {type(problem)}")
        self._problem = problem
        options = options or self.default_options()
        options.validate()
        self._options = options
        self._logger = OptimisationLogger(options.verbose, options.verbose_print_rate)
        self._multistart = options.multistart
        self._set_up_optimiser()

    @staticmethod
    def default_options() -> OptimiserOptions:
        """
        Returns the default options for the optimiser.

        Returns
        -------
        OptimiserOptions
            The default options for the optimiser.
        """
        return OptimiserOptions()

    @property
    def problem(self) -> Problem:
        return self._problem

    @property
    def log(self) -> OptimisationLogger:
        """
        Returns the logger object used for logging optimisation progress.

        Returns
        -------
        OptimisationLogger
            The logger object.
        """
        return self._logger

    @property
    def options(self) -> OptimiserOptions:
        """
        Returns the options for the optimiser.

        Returns
        -------
        OptimiserOptions
            The options for the optimiser.
        """
        return self._options

    @property
    def logger(self) -> OptimisationLogger:
        """
        Returns the logger object used for logging optimisation progress.

        Returns
        -------
        OptimisationLogger
            The logger object.
        """
        return self._logger

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

    def _run(self) -> pybop.OptimisationResult:
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

    def run(self):
        """
        Run the optimisation and return the optimised parameters and final cost.

        Returns
        -------
        results: OptimisationResult
            The pybop optimisation result class.
        """
        results = []
        for i in range(self._multistart):
            if i >= 1:
                initial_values = self.problem.params.rvs(1)[0]
                self.problem.params.update(initial_values=initial_values)
                self._set_up_optimiser()
            results.append(self._run())

        result = pybop.OptimisationResult.combine(results)

        self.problem.params.update(values=result.x_best)

        if self._logger.verbose:
            print(result)

        return result
