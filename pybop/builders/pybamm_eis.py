from collections.abc import Callable

import numpy as np
import pybamm

from pybop import PybammEISProblem, builders
from pybop._pybamm_eis_pipeline import PybammEISPipeline
from pybop.costs.base_cost import BaseCost


class PybammEIS(builders.BaseBuilder):
    def __init__(self):
        super().__init__()
        self.domain = "Frequency [Hz]"

    def set_simulation(
        self,
        model: pybamm.BaseModel,
        parameter_values: pybamm.ParameterValues = None,
        initial_state: tuple = None,
    ) -> None:
        """
        Adds a simulation for the optimisation problem.
        """
        self._model = model.new_copy()
        self._initial_state = initial_state
        self._parameter_values = parameter_values or model.default_parameter_values
        self._solver = pybamm.CasadiSolver()

    def add_cost(self, cost: Callable | BaseCost, weight: float = 1.0) -> None:
        """Adds a cost to the problem."""
        if isinstance(cost, BaseCost):
            if cost.weighting is None or cost.weighting == "equal":
                cost.weighting = 1.0
            elif cost.weighting == "domain":
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

    def build(self) -> PybammEISProblem:
        """
        Builds the Pybamm problem given the provided objects.

        This method requires the following attributes to be set:
            - Dataset
            - Pybamm model
            - Cost(s)
            - Pybop parameters

        Returns
        -------
        Problem : PybammProblem
            A problem instance for optimisation.
        """

        # Checks
        if not len(self._cost_weights) == len(self._costs):
            raise ValueError(
                "Number of cost weights and the number of costs do not match"
            )

        if self._model is None:
            raise ValueError("A Pybamm model needs to be provided before building.")

        if self._costs is None:
            raise ValueError("A cost must be provided before building.")

        if self._dataset is None:
            raise ValueError("A dataset must be provided before building.")

        # Proceed to build the pipeline
        model = self._model
        param = self._parameter_values
        pybop_parameters = self.build_parameters()

        # Build pybamm if not already built
        if not model._built:  # noqa: SLF001
            model.build_model()

        # Construct the pipeline
        pipeline = PybammEISPipeline(
            model,
            self._dataset[self.domain],
            param,
            pybop_parameters,
            self._solver,
        )

        # Build and initialise the pipeline
        pipeline.pybamm_pipeline.build()

        return PybammEISProblem(
            eis_pipeline=pipeline,
            pybop_params=pybop_parameters,
            costs=self._costs,
            cost_weights=self._cost_weights,
            fitting_data=self._dataset["Impedance"],
        )
