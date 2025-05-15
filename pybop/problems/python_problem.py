from typing import Callable

import numpy as np

from pybop import Parameters
from pybop.problems.base_problem import Problem


class PythonProblem(Problem):
    """
    Defines a problem that uses Python callables for simulation and cost evaluation.

    This problem type allows for direct integration with custom Python functions
    instead of relying on external simulation frameworks like PyBaMM.

    Parameters
    ----------
    model_func : Callable
        The function that performs the simulation
    model_func_with_sens : Callable, optional
        Function that performs simulation and returns sensitivities
    pybop_params : Parameters
        Container for optimisation parameters
    costs : list
        Cost objects either Callable | BaseCost
    cost_weights : list
        Weights for each cost component
    """

    def __init__(
        self,
        model_func: Callable,
        model_func_with_sens: Callable = None,
        pybop_params: Parameters = None,
        costs: list[str] = None,
        cost_weights: list[float] = None,
        dataset: dict[str, np.ndarray] = None,
    ):
        super().__init__(pybop_params=pybop_params)
        if not callable(model_func):
            raise TypeError("model_func must be a callable function.")
        if model_func_with_sens is not None and not callable(model_func_with_sens):
            raise TypeError("model_func_with_sens must be a callable function.")

        self._model_func = model_func
        self._model_func_with_sens = model_func_with_sens
        self._costs = costs
        self._cost_weights = cost_weights
        self._dataset = dataset

    def run(self) -> float:
        """
        Evaluates the model and computes the weighted cost.

        Returns
        -------
        float
            The weighted sum of all cost components
        """
        self.check_set_params_called()

        # Run the model with current parameters
        results = self._model_func(self.params, self._dataset)

        costs = [
            cost(results[signal] - self._dataset[column])
            for cost in self._costs
            for column, signal in zip(cost.model_variable, cost.data_column)
        ]
        return np.dot(costs, self._cost_weights)

    def run_with_sensitivities(self) -> tuple[float, np.ndarray]:
        """
        Evaluates the model and returns both cost and sensitivities.

        Returns
        -------
        tuple[float, np.ndarray]
            The cost value and array of parameter gradients
        """
        self.check_set_params_called()

        if self._model_func_with_sens is None:
            raise RuntimeError("No sensitivity function provided")

        # Run the model with current parameters
        results, sens = self._model_func_with_sens(
            self.params, self._dataset
        )  # ToDo: sense needs a better format (combined dict.sens?, or just dict)

        costs = [
            cost(results[signal] - self._dataset[column], dy=sens)
            for cost in self._costs
            for column, signal in zip(cost.model_variable, cost.data_column)
        ]

        return np.dot(costs, self._cost_weights)

    def set_params(self, p: np.ndarray) -> None:
        """
        Sets the parameter values for simulation.

        Parameters
        ----------
        p : np.ndarray
            Array of parameter values
        """
        self.check_and_store_params(p)
