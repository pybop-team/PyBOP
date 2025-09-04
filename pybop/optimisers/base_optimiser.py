from dataclasses import dataclass

import pybop
from pybop.problems.base_problem import Problem


@dataclass
class OptimiserOptions:
    """
    A base class for optimiser options.

    Attributes:
        multistart (int): Number of times to multistart the optimiser
        verbose (bool): The verbosity level
        verbose_print_rate (int): The distance between iterations to print verbose output
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
        An objective function to be optimised.
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
        self.verbose = options.verbose
        self.verbose_print_rate = options.verbose_print_rate
        self._multistart = options.multistart
        self._set_up_optimiser()

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

    def name(self) -> str:
        """
        Returns the name of the optimiser, to be overwritten by child classes.

        Returns
        -------
        str
            The name of the optimiser
        """
        raise NotImplementedError  # pragma: no cover

    def run(self) -> pybop.OptimisationResult:
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
                if not self.problem.params.priors():
                    raise RuntimeError("Priors must be provided for multi-start")
                initial_values = self.problem.params.sample_from_priors(1)[0]
                self.problem.params.update(initial_values=initial_values)
                self._set_up_optimiser()
            results.append(self._run())

        result = pybop.OptimisationResult.combine(results)

        self.problem.params.update(values=result.x)

        if self.options.verbose:
            print(result)

        return result

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
        return pybop.plot.convergence(self, **kwargs)

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
        return pybop.plot.parameters(self, **kwargs)

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
        return pybop.plot.surface(self, **kwargs)

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
        return pybop.plot.contour(self, **kwargs)
