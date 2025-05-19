from typing import Union

import pybamm

import pybop
from pybop._pybamm_pipeline import PybammPipeline
from pybop.costs.pybamm_cost import PybammCost

from .base import BaseBuilder


class Pybamm(BaseBuilder):
    def __init__(self):
        self._model = None
        self._costs = []
        self._dataset = None
        self._solver = None
        self._parameter_values = None
        self._rebuild_parameters = None
        self._initial_state = None
        self._pipeline = None
        self.domain = "Time [s]"
        super().__init__()

    def set_simulation(
        self,
        model: pybamm.BaseModel,
        parameter_values: pybamm.ParameterValues = None,
        solver: pybamm.BaseSolver = None,
        initial_state: Union[float, str] = None,
    ) -> None:
        """
        Adds a simulation for the optimisation problem.
        """
        self._model = model.new_copy()
        self._initial_state = initial_state
        self._parameter_values = parameter_values or model.default_parameter_values
        self._solver = solver or model.default_solver

    def add_cost(self, cost: PybammCost, weight: float = 1.0) -> None:
        self._costs.append(cost)
        self._cost_weights.append(weight)

    def build(self) -> pybop.PybammProblem:
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

        # Proceed to building the pipeline
        model = self._model
        pybamm_parameter_values = self._parameter_values
        pybop_parameters = self.build_parameters()

        # Build pybamm if not already built
        if not model._built:  # noqa: SLF001
            model.build_model()

        # Set the control variable
        if self._dataset is not None:
            self._set_control_variable(pybop_parameters)

        # add costs
        cost_names = []
        for cost in self._costs:
            cost.add_to_model(model, pybamm_parameter_values, self._dataset)
            cost_names.append(cost.metadata().variable_name)

        # Construct the pipeline
        pipeline = PybammPipeline(
            model,
            pybamm_parameter_values,
            pybop_parameters,
            self._solver,
            t_start=self._dataset[self.domain][0],
            t_end=self._dataset[self.domain][-1],
            t_interp=self._dataset[self.domain],
            initial_state=self._initial_state,
        )

        # Build the pipeline
        pipeline.build()

        return pybop.PybammProblem(
            pybamm_pipeline=pipeline,
            pybop_params=pybop_parameters,
            cost_names=cost_names,
            cost_weights=self._cost_weights,
        )

    def _set_control_variable(self, pybop_parameters: pybop.Parameters) -> None:
        """
        Updates the pybamm parameter values to match the control variable
        time-series. This is conventionally the applied current; however,
        alternative control methods are supported.
        """
        control = (
            self._dataset.control_variable
        )  # Add a control attr to dataset w/ catches
        if control in self._parameter_values:
            if control not in pybop_parameters:
                control_interpolant = pybamm.Interpolant(
                    self._dataset["Time [s]"],
                    self._dataset[control],
                    pybamm.t,
                )
                if control == "Current [A]":
                    self._parameter_values["Current function [A]"] = control_interpolant
                else:
                    self._parameter_values[control] = control_interpolant
