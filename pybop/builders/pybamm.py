import pybamm

from pybop import Parameter, Parameters, builders
from pybop._pybamm_pipeline import PybammPipeline
from pybop.costs.pybamm_cost import PybammCost
from pybop.problems.pybamm_problem import PybammProblem


class Pybamm(builders.BaseBuilder):
    def __init__(self):
        self._pybamm_models = []
        self._costs = []
        self._cost_names = []
        self._dataset = None
        self._pybop_parameters = Parameters()
        self._solver = []
        self._parameter_values = []
        self._rebuild_parameters = []
        self._simulation_weights = []
        self._cost_weights = []
        self._pipeline = []

    def add_simulation(
        self,
        pybamm_model: pybamm.BaseModel,
        parameter_values: pybamm.ParameterValues = None,
        solver: pybamm.BaseSolver = None,
        weight: float = 1.0,
    ) -> None:
        """
        Adds a simulation for the optimisation problem.
        """
        self._pybamm_models.append(pybamm_model.new_copy())
        self._parameter_values.append(
            parameter_values or pybamm_model.default_parameter_values
        )
        self._solver.append(solver or pybamm_model.default_solver)
        self._simulation_weights.append(weight)

    def add_cost(self, cost: PybammCost, weight: float = 1.0) -> None:
        self._costs.append(cost)
        self._cost_names.append(cost.variable_name())
        self._cost_weights.append(weight)

    def add_parameter(self, parameter: Parameter) -> None:
        self._pybop_parameters.add(parameter)

    def _requires_rebuild(self, built_model: pybamm.BaseModel) -> bool:
        solver = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-3)
        try:
            solver.solve(built_model, t_eval=[0, 1])
        except pybamm.SolverError:  # Change to ValueError (i3649)
            # If the solver fails, it indicates that the model needs to be rebuilt
            return True
        return False

    def build(self) -> PybammProblem:
        """
        Build the Pybamm Problem.
        """

        # Checks
        if not len(self._simulation_weights) == len(self._pybamm_models):
            raise ValueError(
                "Number of simulation weights and the number of pybamm models do not match"
            )

        if not len(self._cost_weights) == len(self._costs):
            raise ValueError(
                "Number of cost weights and the number of costs do not match"
            )

        # Proceed to building the pipeline(s)
        for i, pybamm_model in enumerate(self._pybamm_models):
            model = pybamm_model.new_copy()
            param = self._parameter_values[i]

            # Build pybamm if not already built
            if not model._built:  # noqa: SLF001
                model.build_model()

            # Set the control variable
            if self._dataset is not None:
                self._set_control_variable()

            # add costs
            for cost in self._costs:
                cost.add_to_model(model, param)

            # set input parameters
            for parameter in self._pybop_parameters:
                param.update({parameter.name: "[input]"})

            # Construct the pipeline
            pipeline = PybammPipeline(
                model,
                param,
                self._solver[i],
            )

            # Build the pipeline, determine if the parameters require rebuilding
            pipeline.build()
            requires_rebuild = self._requires_rebuild(pipeline.built_model)

            # Add to the parameter names attr if rebuild required
            if requires_rebuild:
                pipeline.parameter_names = self._pybop_parameters.keys()
            self._pipeline.append(pipeline)

        return PybammProblem(
            pybamm_pipeline=self._pipeline,
            param_names=self._pybop_parameters.keys(),
            cost_names=self._cost_names,
            simulation_weights=self._simulation_weights,
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
        for parameter_values in self._parameter_values:
            if control in parameter_values:
                if control not in self._pybop_parameters.keys():
                    control_interpolant = pybamm.Interpolant(
                        self._dataset["Time [s]"],
                        self._dataset[control],
                        pybamm.t,
                    )
                    if control == "Current [A]":
                        parameter_values["Current function [A]"] = control_interpolant
                    else:
                        parameter_values[control] = control_interpolant
