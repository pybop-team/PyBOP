import pybop
from pybop import PythonProblem
from pybop.builders import BaseBuilder
from pybop.problems.base_problem import Problem


class Python(BaseBuilder):
    """
    dataset : pybop.Dataset or dict, optional
        The dataset to be used in the simulation construction.
    """

    def __init__(self):
        self._model = None
        self._params = None
        self._costs = None
        self._cost_weights = []

    def add_func(self, func):
        self._func = func

    def add_cost(self, cost: pybop.BaseCost, weight: float = 1.0) -> None:
        """
        Adds cost to the problem.
        ToDo: BaseCost needs to be updated to support array/list-like execution
        """
        self._costs.append(cost)
        self._cost_weights.append(weight)

    def build(self) -> Problem:
        """Build a pure python problem."""

        if not len(self._cost_weights) == len(self._costs):
            raise ValueError(
                "Number of cost weights and the number of costs do not match"
            )

        if self._func is None:
            raise ValueError("A Pybamm model needs to be provided before building.")

        if self._costs is None:
            raise ValueError("A cost must be provided before building.")

        if self._dataset is None:
            raise ValueError("A dataset must be provided before building.")

        return PythonProblem(
            self._func,
        )
