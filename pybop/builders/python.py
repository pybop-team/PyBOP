from typing import Callable, Union

import numpy as np

from pybop import PythonProblem
from pybop.builders.base import BaseBuilder
from pybop.costs.base_cost import BaseCost


class Python(BaseBuilder):
    """
    Builder for Python-based optimisation problems.

    This builder allows for the creation of optimisation problems using custom
    Python functions instead of specialised simulation frameworks.

    """

    def __init__(self):
        super().__init__()
        self._model = None
        self._model_with_sens = None

    def set_simulation(
        self,
        model: Callable = None,
        model_with_sens: Callable = None,
    ) -> None:
        """
        Set the simulation functions for the problem.

        Parameters
        ----------
        model : Callable
            Function that takes parameters and dataset, returns simulation results
            Expected signature: func(params: dict, dataset: dict) -> dict
        model_with_sens : Callable, optional
            Function that returns both results and sensitivities
            Expected signature: func(params: dict, dataset: dict) -> tuple[dict, np.ndarray]
        """
        if model is not None and not callable(model):
            raise TypeError("The model must be a callable obj")

        if model_with_sens is not None and not callable(model_with_sens):
            raise TypeError("The model must be a callable obj")

        self._model = model
        self._model_with_sens = model_with_sens

    def add_cost(self, cost: Union[BaseCost, Callable], weight: float = 1.0) -> None:
        """
        Add a cost component to the problem.

        Parameters
        ----------
        cost : Union[BaseCost, Callable]
            Cost function
        weight : float, optional
            Weight for this cost component, by default 1.0
        """
        if isinstance(cost, BaseCost):
            if cost.weighting is None or cost.weighting == "equal":
                cost.weighting = 1.0
            elif cost.weighting == "domain" and self.domain is not None:
                self._set_cost_domain_weighting(cost)

        self._costs.append(cost)
        self._cost_weights.append(weight)

    def _set_cost_domain_weighting(self, cost):
        """Calculate domain-based weighting."""
        domain_data = self._dataset[self.domain]
        domain_spacing = domain_data[1:] - domain_data[:-1]
        mean_spacing = np.mean(domain_spacing)

        # Create a domain weighting array in one operation
        cost.weighting = np.concatenate(
            (
                [(mean_spacing + domain_spacing[0]) / 2],
                (domain_spacing[1:] + domain_spacing[:-1]) / 2,
                [(domain_spacing[-1] + mean_spacing) / 2],
            )
        ) * ((len(domain_data) - 1) / (domain_data[-1] - domain_data[0]))

    def build(self) -> PythonProblem:
        """
        Build the Python problem with the configured components.

        Returns
        -------
        PythonProblem
            The constructed optimisation problem

        Raises
        ------
        ValueError
            If required components are missing
        """
        # Validate required components
        if self._model is None and self._model_with_sens is None:
            raise ValueError("A model function must be provided before building")

        if not self._costs:
            raise ValueError("At least one cost must be provided before building")

        if self._dataset is None:
            raise ValueError("A dataset must be provided before building")

        if len(self._costs) != len(self._cost_weights):
            raise ValueError("Number of cost weights and costs do not match")

        pybop_parameters = self.build_parameters()

        # Create and return the problem
        return PythonProblem(
            model=self._model,
            model_with_sens=self._model_with_sens,
            pybop_params=pybop_parameters,
            costs=self._costs,
            cost_weights=self._cost_weights,
            dataset=self._dataset.as_dict()
            if hasattr(self._dataset, "as_dict")
            else self._dataset,
        )
