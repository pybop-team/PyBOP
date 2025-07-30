from pybop import Parameters, Problem, PythonProblem
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

        funs = []
        funs_with_sense = []
        # Construct the list of callables
        for problem in self._problems:
            funs.append(lambda params, p=problem: (p.set_params(params), p.run())[1])
            funs_with_sense.append(
                lambda params, p=problem: (
                    p.set_params(params),
                    p.run_with_sensitivities(),
                )[1]
            )

        return PythonProblem(
            funs=funs,
            funs_with_sens=funs_with_sense,
            pybop_params=self.build_parameters(),
            weights=self._weights,
        )

    def __len__(self) -> int:
        """Return the number of problems."""
        return len(self._problems)
