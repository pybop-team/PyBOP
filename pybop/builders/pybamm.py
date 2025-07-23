import pybamm

import pybop
from pybop import Parameter as PybopParameter
from pybop._pybamm_pipeline import PybammPipeline
from pybop.builders.base import BaseBuilder
from pybop.builders.utils import cell_mass, set_formation_concentrations
from pybop.costs.pybamm import BaseLikelihood, DesignCost, PybammCost


class TIME_PARAMS:
    """Enum-like class for time params"""

    time_params = {"t_eval": None, "t_interp": None}


class Pybamm(BaseBuilder):
    def __init__(self):
        self._model: pybamm.BaseModel | None = None
        self._solver: pybamm.BaseSolver | None = None
        self._parameter_values: pybamm.ParameterValues | None = None
        self._initial_state: float | str | None = None
        self._experiment: pybamm.Experiment | None = None
        self._costs: list[PybammCost] = []
        self._cost_weights: list[float] = []
        self.domain = "Time [s]"
        self.is_posterior = False
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
        self._solver = solver or model.default_solver
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

    def remove_costs(self) -> None:
        self._costs = []
        self._cost_weights = []

    def set_experiment(self, experiment: pybamm.Experiment) -> None:
        self._experiment = experiment

    def build(self) -> pybop.PybammProblem:
        """
        Builds the Pybamm problem given the provided objects.

        This method requires the following attributes to be set:
            - Dataset | Experiment
            - Pybamm model
            - Cost(s)
            - Pybop parameters

        Returns
        -------
        Problem : PybammProblem
            A problem instance for optimisation.
        """

        # Checks
        self._validate_build_requirements()

        # Proceed to building the pipeline
        model = self._model
        pybamm_parameter_values = self._parameter_values
        pybop_parameters = self.build_parameters()
        time_params = TIME_PARAMS.time_params

        # Build pybamm if not already built
        if not model.built:
            model.build_model()

        # Set the control variable
        if self._dataset is not None:
            self._set_control_variable(pybop_parameters)
            time_params = self._extract_time_parameters()

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
                self.is_posterior = True

            # Add hypers to pybop parameters
            if cost.metadata().parameters:
                for name, obj in cost.metadata().parameters.items():
                    delta = obj.default_value * 0.5  # Create prior w/ large variance
                    prior = (
                        pybop.Gaussian(obj.default_value, delta)
                        if self.is_posterior
                        else None
                    )
                    pybop_parameters.add(
                        PybopParameter(
                            name, initial_value=obj.default_value, prior=prior
                        )
                    )

            # Posterior Flag
            if isinstance(cost, BaseLikelihood) and pybop_parameters.priors():
                self.is_posterior = True

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
            experiment=self._experiment,
            t_eval=time_params["t_eval"],
            t_interp=time_params["t_interp"],
            initial_state=self._initial_state,
            build_on_eval=self._build_on_eval,
        )

        # Build the pipeline
        pipeline.build()

        return pybop.PybammProblem(
            pybamm_pipeline=pipeline,
            pybop_params=pybop_parameters,
            cost_names=cost_names,
            cost_weights=self._cost_weights,
            is_posterior=self.is_posterior,
            use_last_cost_index=use_last_index,
        )

    def _extract_time_parameters(self) -> dict:
        """Extract time-related parameters from dataset."""
        domain_data = self._dataset[self.domain]
        return {
            "t_eval": [domain_data[0], domain_data[-1]],
            "t_interp": domain_data,
        }

    def _validate_build_requirements(self) -> None:
        """Validate all required components are set before building."""
        if len(self._cost_weights) != len(self._costs):
            raise ValueError("Number of cost weights and costs do not match")

        if self._model is None:
            raise ValueError("A Pybamm model needs to be provided before building")

        if not self._costs:
            raise ValueError("A cost must be provided before building")

        if self._experiment is None and self._dataset is None:
            raise ValueError("A dataset must be provided before building")

    def _set_control_variable(self, pybop_parameters: pybop.Parameters) -> None:
        """
        Updates the pybamm parameter values to match the control variable
        time-series. This is conventionally the applied current; however,
        alternative control methods are supported.
        """
        control = self._dataset.control_variable

        if control not in self._parameter_values or control in pybop_parameters:
            return

        control_interpolant = pybamm.Interpolant(
            self._dataset["Time [s]"],
            self._dataset[control],
            pybamm.t,
        )

        # Handle special case for current
        param_key = "Current function [A]" if control == "Current [A]" else control
        self._parameter_values[param_key] = control_interpolant
