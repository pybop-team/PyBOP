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
        self._parameters = Parameters()
        self._solver = []
        self._parameter_values = []
        self._rebuild_parameters = []
        self._pipeline = []

    def add_simulation(
        self,
        pybamm_model: pybamm.BaseModel,
        parameter_values: pybamm.ParameterValues = None,
        solver: pybamm.BaseSolver = None,
    ) -> None:
        """
        Adds a simulation for the optimisation problem.
        """
        self._pybamm_models.append(pybamm_model.new_copy())
        self._parameter_values.append(
            parameter_values or pybamm_model.default_parameter_values
        )
        self._solver.append(solver or pybamm_model.default_solver)

    def add_cost(self, cost: PybammCost) -> None:
        self._costs.append(cost)
        self._cost_names.append(cost.variable_name())

    def add_parameter(self, parameter: Parameter) -> None:
        self._parameters.add(parameter)

    def _requires_rebuild(self, built_model: pybamm.BaseModel) -> bool:
        solver = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-3)
        try:
            solver.solve(built_model, t_eval=[0, 1])
        except pybamm.SolverError:  # Change to ValueError (i3649)
            # If the solver fails, it indicates that the model needs to be rebuilt
            return True
        return False

    def build(self) -> PybammProblem:
        for i, pybamm_model in enumerate(self._pybamm_models):
            model = pybamm_model.deep_copy()
            param = self._parameter_values[i]

            if not model._built:  # noqa: SLF001
                model.build_model()

            if self._dataset is not None:
                self.set_current_function()

            # add costs
            for cost in self._costs:
                cost.add_to_pybamm_model(model, param)

            # set input parameters
            for parameter in self._parameters:
                param.update(parameter.name, "[input]")

            requires_rebuild = self._requires_rebuild(model)

            if requires_rebuild:
                self.rebuild_parameters = (
                    self._parameters.keys()
                )  # Needs to be parameters object

            self._pipeline.append(
                PybammPipeline(
                    model,
                    param,
                    self._solver[i],
                    rebuild_parameters=self._rebuild_parameters,
                )
            )

            return PybammProblem(
                pybamm_pipeline=self._pipeline,
                param_names=self._parameters.keys(),
                cost_names=self._cost_names,
            )

    def set_current_function(self) -> None:
        """
        Update the input current function according to the data.

        Parameters
        ----------
        dataset : pybop.Dataset or dict, optional
            The dataset to be used in the model construction.
        """
        if "Current function [A]" in self._parameter_values:
            if "Current function [A]" not in self._parameters.keys():
                current = pybamm.Interpolant(
                    self._dataset["Time [s]"],
                    self._dataset["Current function [A]"],
                    pybamm.t,
                )
                self._parameter_values["Current function [A]"] = current
