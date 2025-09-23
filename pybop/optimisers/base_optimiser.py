from dataclasses import dataclass

from pybop._logging import Logger
from pybop._result import OptimisationResult
from pybop.problems.base_problem import Problem


@dataclass
class OptimiserOptions:
    """
    A base class for optimiser options.

    Attributes
    ----------
    multistart : int
        Number of times to multistart the optimiser.
    verbose : bool
        The verbosity level.
    verbose_print_rate : int
        The distance between iterations to print verbose output.
    """

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
        The problem to optimise.
    options: pybop.OptimiserOptions , optional
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
        self._problem.parameters.reset_to_initial()
        self._logger = None
        options = options or self.default_options()
        options.validate()
        self._options = options
        self.verbose = options.verbose
        self.verbose_print_rate = options.verbose_print_rate
        self._multistart = options.multistart
        self._needs_sensitivities = None  # to be overridden during set_up_optimiser
        self._set_up_optimiser()
        if self._needs_sensitivities and not self._problem.has_sensitivities:
            raise ValueError(
                "This optimiser needs sensitivities, but they are not available from this problem."
            )

    @staticmethod
    def default_options() -> OptimiserOptions:
        """Returns the default options for the optimiser."""
        return OptimiserOptions()

    @property
    def problem(self) -> Problem:
        """Returns the optimisation problem object."""
        return self._problem

    @property
    def options(self) -> OptimiserOptions:
        """Returns the options for the optimiser."""
        return self._options

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

    def _run(self) -> OptimisationResult:
        """
        Contains the logic for the optimisation algorithm.

        This method should be implemented by child classes to perform the actual optimisation.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the subclass.
        """
        raise NotImplementedError

    def name(self) -> str:
        """
        Returns the name of the optimiser, to be overwritten by child classes.

        Returns
        -------
        str
            The name of the optimiser
        """
        raise NotImplementedError  # pragma: no cover

    def run(self) -> OptimisationResult:
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
                if not self.problem.parameters.priors():
                    raise RuntimeError("Priors must be provided for multi-start")
                initial_values = self.problem.parameters.sample_from_priors(1)[0]
                self.problem.parameters.update(initial_values=initial_values)
                self._set_up_optimiser()
            results.append(self._run())

        result = OptimisationResult.combine(results)

        self.problem.parameters.update(values=result.x)

        if self.options.verbose:
            print(result)

        return result

    @property
    def logger(self) -> Logger | None:
        return self._logger
