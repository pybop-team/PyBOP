from typing import Callable

import numpy as np

from pybop import Parameters
from pybop.costs.base_cost import BaseCost
from pybop.problems.base_problem import Problem


class PythonProblem(Problem):
    """
    Defines a problem that uses Python callables for simulation and cost evaluation.

    This problem type allows for direct integration with custom Python functions
    instead of relying on external simulation frameworks like PyBaMM.

    Parameters
    ----------
    model : Callable
        The function that performs the simulation
    model_with_sens : Callable, optional
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
        model: Callable,
        model_with_sens: Callable = None,
        pybop_params: Parameters = None,
        costs: list[BaseCost] = None,
        cost_weights: list[float] = None,
        dataset: dict[str, np.ndarray] = None,
    ):
        super().__init__(pybop_params=pybop_params)
        self._model = model
        self._model_with_sens = model_with_sens
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
        results = self._model(self.params.current_value())

        total_cost: float = 0.0

        # Calculate weighted costs and gradients
        for i, cost_func in enumerate(self._costs):
            if hasattr(cost_func, "model_variable") and hasattr(
                cost_func, "data_column"
            ):
                for column, signal in zip(
                    cost_func.data_column, cost_func.model_variable
                ):
                    if signal not in results or column not in self._dataset:
                        raise KeyError(
                            f"Signal '{signal}' or column '{column}' not found"
                        )

                    residual = results[signal] - self._dataset[column]
                    val = cost_func(residual)
                    total_cost += val * self._cost_weights[i]
            else:
                raise RuntimeError(
                    "Cost function must have model_variable and data_column"
                )

        return total_cost

    def run_with_sensitivities(self) -> tuple[float, np.ndarray]:
        """
        Evaluates the model and returns both cost and sensitivities.

        Returns
        -------
        tuple[float, np.ndarray]
            The cost value and array of parameter gradients
        """
        self.check_set_params_called()
        if self._model_with_sens is None:
            raise RuntimeError("No sensitivity function provided")

        # Run the model with current parameters
        results, sens = self._model_with_sens(self.params.current_value())

        if results is None or sens is None:
            raise RuntimeError("Model returned empty results or sensitivities")

        total_cost = 0.0
        total_grad = np.zeros(len(self.params))

        # Calculate weighted costs and gradients
        for i, cost_func in enumerate(self._costs):
            if hasattr(cost_func, "model_variable") and hasattr(
                cost_func, "data_column"
            ):
                for column, signal in zip(
                    cost_func.data_column, cost_func.model_variable
                ):
                    if signal not in results or column not in self._dataset:
                        raise KeyError(
                            f"Signal '{signal}' or column '{column}' not found"
                        )
                    if signal not in sens:
                        raise KeyError(f"Sensitivity for '{signal}' not found")

                    residual = results[signal] - self._dataset[column]
                    val, grad = cost_func(residual, dy=sens[signal])
                    total_cost += val * self._cost_weights[i]
                    total_grad += grad * self._cost_weights[i]
            else:
                raise RuntimeError(
                    "Cost function must have model_variable and data_column"
                )

        return total_cost, total_grad

    def set_params(self, p: np.ndarray) -> None:
        """
        Sets the parameter values for simulation.

        Parameters
        ----------
        p : np.ndarray
            Array of parameter values
        """
        self.check_and_store_params(p)
