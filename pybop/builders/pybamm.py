import pybamm

import pybop
from pybop import Parameter as PybopParameter
from pybop._pybamm_pipeline import PybammPipeline
from pybop.builders.base import BaseBuilder
from pybop.builders.utils import cell_mass, set_formation_concentrations
from pybop.costs.pybamm import BaseLikelihood, DesignCost, PybammCost


class Pybamm(BaseBuilder):
    def __init__(self):
        self._model = None
        self._solver = None
        self._parameter_values = None
        self._rebuild_parameters = None
        self._initial_state = None
        self._pipeline = None
        self._costs: list[PybammCost] = []
        self._cost_weights: list[float] = []
        self.domain = "Time [s]"
        self._use_posterior = False
        super().__init__()

    def set_simulation(
        self,
        model: pybamm.BaseModel,
        parameter_values: pybamm.ParameterValues = None,
        solver: pybamm.BaseSolver = None,
        initial_state: float | str = None,
        build_on_eval: bool = None,
    ) -> None:
        """
        Adds a simulation for the optimisation problem.

        Parameters
        ----------
        model : pybamm.BaseModel
            The PyBaMM model to be used.
        parameter_values : pybamm.ParameterValues
            The parameters to be used in the model.
        solver : pybamm.BaseSolver
            The solver to be used. If None, the idaklu solver will be used.
        initial_state: float | str
            The initial state of charge or voltage for the battery model. If float, it will be represented
            as SoC and must be in range 0 to 1. If str, it will be represented as voltage and needs to be in
            the format: "3.4 V".
        build_on_eval : bool
            Boolean to determine if the model will be rebuilt every evaluation. If `initial_state` is provided,
            the model will be rebuilt every evaluation unless `build_on_eval` is `False`, in which case the model
            is built with the parameter values from construction only.
        """
        self._model = model.new_copy()
        self._initial_state = initial_state
        self._build_on_eval = build_on_eval
        self._solver = solver
        self._parameter_values = (
            parameter_values.copy()
            if parameter_values
            else model.default_parameter_values
        )

    def add_cost(self, cost: PybammCost, weight: float = 1.0) -> None:
        """
        Adds a cost to the problem with optional weighting.
        """
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
        use_last_index = []
        for cost in self._costs:
            cost.add_to_model(model, pybamm_parameter_values, self._dataset)
            cost_names.append(cost.metadata().variable_name)
            use_last_index.append(
                isinstance(cost.metadata().expression, pybamm.ExplicitTimeIntegral)
            )

            # Posterior Logic
            if isinstance(cost, BaseLikelihood) and pybop_parameters.priors():
                self._use_posterior = True

            # Add hypers to pybop parameters
            if cost.metadata().parameters:
                for name, obj in cost.metadata().parameters.items():
                    delta = obj.default_value * 0.5  # Create prior w/ large variance
                    prior = (
                        pybop.Gaussian(obj.default_value, delta)
                        if self._use_posterior
                        else None
                    )
                    pybop_parameters.add(
                        PybopParameter(
                            name, initial_value=obj.default_value, prior=prior
                        )
                    )

            # Design Costs
            if isinstance(cost, DesignCost):
                cell_mass(pybamm_parameter_values)
                set_formation_concentrations(pybamm_parameter_values)

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
            build_on_eval=self._build_on_eval,
            cost_names=cost_names,
        )

        # Build the pipeline
        pipeline.build()

        return pybop.PybammProblem(
            pybamm_pipeline=pipeline,
            pybop_params=pybop_parameters,
            cost_names=cost_names,
            cost_weights=self._cost_weights,
            use_posterior=self._use_posterior,
            use_last_cost_index=use_last_index,
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
