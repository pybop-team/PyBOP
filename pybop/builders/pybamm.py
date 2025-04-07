import pybamm

from pybop import Parameter, Parameters, builders
from pybop._pybamm_pipeline import PybammPipeline
from pybop.costs.pybamm_cost import PybammCost
from pybop.problems.pybamm_problem import PybammProblem


class Pybamm(builders.BaseBuilder):
    def __init__(self):
        super().__init__()
        self._pybamm_model = None
        self._costs = []
        self._cost_names = []
        self._dataset = None
        self._parameters = Parameters()
        self._solver = None
        self._parameter_values = None
        self._rebuild_parameters = []

    def set_simulation(
        self,
        pybamm_model: pybamm.BaseModel,
        parameter_values: pybamm.ParameterValues = None,
        solver: pybamm.BaseSolver = None,
    ) -> None:
        """
        Sets the simulation for the optimisation problem.

        :param pybamm_model:
        :param parameter_values:
        :param solver:
        :return:
        """
        self._pybamm_model = pybamm_model.new_copy()
        self._parameter_values = (
            parameter_values or self._pybamm_model.default_parameter_values
        )
        self._solver = solver or self._pybamm_model.default_solver

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
        model = self._pybamm_model.deep_copy()
        param = self._parameter_values

        if not self._pybamm_model._built:  # noqa: SLF001
            self._pybamm_model.build_model()

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

        self._pipeline = PybammPipeline(
            self._pybamm_model,
            self._parameter_values,
            self._solver,
            rebuild_parameters=self._rebuild_parameters,
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
