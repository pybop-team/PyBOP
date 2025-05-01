import pybamm

from pybop import Parameter, Parameters, PybammProblem, builders
from pybop._pybamm_pipeline import PybammPipeline
from pybop.costs.pybamm_cost import PybammCost


class Pybamm(builders.BaseBuilder):
    def __init__(self):
        self._pybamm_model = None
        self._costs = []
        self._cost_names = []
        self._dataset = None
        self._pybop_parameters = Parameters()
        self._solver = None
        self._parameter_values = None
        self._rebuild_parameters = None
        self._cost_weights = []
        self._pipeline = None

    def set_simulation(
        self,
        pybamm_model: pybamm.BaseModel,
        parameter_values: pybamm.ParameterValues = None,
        solver: pybamm.BaseSolver = None,
    ) -> None:
        """
        Adds a simulation for the optimisation problem.
        """
        self._pybamm_model = pybamm_model.new_copy()
        self._parameter_values = (
            parameter_values or pybamm_model.default_parameter_values
        )
        self._solver = solver or pybamm_model.default_solver

    def add_cost(self, cost: PybammCost, weight: float = 1.0) -> None:
        self._costs.append(cost)
        self._cost_names.append(cost.variable_name())
        self._cost_weights.append(weight)

    def add_parameter(self, parameter: Parameter) -> None:
        self._pybop_parameters.add(parameter)

    def build(self) -> PybammProblem:
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

        if self._pybamm_model is None:
            raise ValueError("A Pybamm model needs to be provided before building.")

        if self._costs is None:
            raise ValueError("A cost must be provided before building.")

        if self._dataset is None:
            raise ValueError("A dataset must be provided before building.")

        # Proceed to building the pipeline
        model = self._pybamm_model
        param = self._parameter_values

        # Build pybamm if not already built
        if not model._built:  # noqa: SLF001
            model.build_model()

        # Set the control variable
        if self._dataset is not None:
            self._set_control_variable()

        # add costs
        for cost in self._costs:
            cost.add_to_model(model, param)

        # Construct the pipeline
        pipeline = PybammPipeline(
            model,
            param,
            self._solver,
        )

        if not pipeline.requires_rebuild:
            # set input parameters
            for parameter in self._pybop_parameters:
                param.update({parameter.name: "[input]"})

        # Build the pipeline, determine if the parameters require rebuilding
        pipeline.build()

        # Add to the parameter names attr if rebuild required
        if pipeline.requires_rebuild:
            pipeline.parameter_names = self._pybop_parameters.keys()

        return PybammProblem(
            pybamm_pipeline=pipeline,
            pybop_params=self._pybop_parameters,
            cost_names=self._cost_names,
            cost_weights=self._cost_weights,
        )

    def _set_control_variable(self) -> None:
        """
        Updates the pybamm parameter values to match the control variable
        time-series. This is conventionally the applied current; however,
        alternative control methods are supported.
        """
        control = (
            self._dataset.control_variable
        )  # Add a control attr to dataset w/ catches
        if control in self._parameter_values:
            if control not in self._pybop_parameters.keys():
                control_interpolant = pybamm.Interpolant(
                    self._dataset["Time [s]"],
                    self._dataset[control],
                    pybamm.t,
                )
                if control == "Current [A]":
                    self._parameter_values["Current function [A]"] = control_interpolant
                else:
                    self._parameter_values[control] = control_interpolant
