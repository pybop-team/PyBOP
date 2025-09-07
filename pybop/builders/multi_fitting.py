import numpy as np

from pybop import Inputs, Parameters, Problem, PythonProblem
from pybop.builders.base import BaseBuilder


class MultiFitting(BaseBuilder):
    """
    Builder for multiple problems.

    If this problem is used by Pybop Optimisation or Sampling classes, the
    problems it is constructed with will be minimised.

    This builder constructs a PythonProblem from the provided functions.

    Examples
    --------
    >>> builder = pybop.builders.MultiFitting()
    >>> builder.add_problem(problem1, weight=1.5)
    >>> builder.add_problem(problem2)
    >>> problem = builder.build()
    """

    def __init__(self):
        super().__init__()
        self._problems: list[Problem] = []
        self._weights: list[float] = []
        self._parameters: list[Parameters] = []

    def add_problem(self, problem: Problem, weight: float = 1.0) -> "MultiFitting":
        """
        Add a problem to the builder.

        Parameters
        ----------
        Problem: pybop.Problem
            A problem to be added, built from a previous pybop builder.

        Returns
        -------
        Python
            Self for method chaining.

        Raises
        ------
        TypeError
            If model is not callable.
        """
        if not isinstance(problem, Problem):
            raise TypeError("Problem must be an instance of pybop.Problem")

        self._parameters = set(problem.params.keys())
        if len(self._parameters) != len(problem.params):
            raise TypeError("All problems must have the same parameters")

        self._problems.append(problem)
        self._weights.append(weight)

        return self

    def remove_problems(self) -> "MultiFitting":
        self._problems.clear()
        self._weights.clear()
        self._parameters.clear()

        return self

    def build(self) -> PythonProblem:
        """
        Build the Python problem.

        Returns
        -------
        PythonProblem
            The constructed problem with all configured components.

        Raises
        ------
        ValueError
            If no models are provided or if both model types are specified.
        """
        if not self._problems:
            raise ValueError("At least one problem must be added before building")

        # TODO:
        # Check bounds and conform to the tighter bounds
        # Compare priors and initial conditions and select
        self._params = self._problems[0].params.copy()

        n_problems = len(self._problems)
        weights = (
            np.asarray(self._weights)
            if self._weights is not None and len(self._weights) > 0
            else np.ones(n_problems)
        )

        def cost(inputs: list[Inputs]) -> np.ndarray:
            costs = np.empty((len(inputs), n_problems))
            for i, problem in enumerate(self._problems):
                costs[:, i] = problem._compute_costs([inputs])  # noqa: SLF001
            return np.matmul(costs, weights)

        cost_with_sens = None
        if all([p.has_sensitivities for p in self._problems]):

            def cost_with_sens(inputs: list[Inputs]) -> tuple[np.ndarray, np.ndarray]:
                costs = np.empty((len(inputs), n_problems))
                sens = np.empty((len(inputs), len(self._params), n_problems))
                for i, problem in enumerate(self._problems):
                    results = problem._compute_costs_and_sensitivities([inputs])  # noqa: SLF001
                    costs[:, i] = results[0]
                    sens[:, :, i] = results[1]
                return np.matmul(costs, weights), np.matmul(sens, weights)

        return PythonProblem(
            cost=cost,
            cost_with_sens=cost_with_sens,
            vectorised=True,
            pybop_params=self.build_parameters(),
        )

    def __len__(self) -> int:
        """Return the number of problems."""
        return len(self._problems)
