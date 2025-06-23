import pybamm

import pybop
from pybop import Parameter as PybopParameter
from pybop._pybamm_pipeline import PybammPipeline
from pybop.builders.base import BaseBuilder
from pybop.costs.pybamm import PybammCost, BaseLikelihood, DesignCost


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
    ) -> None:
        """
        Adds a simulation for the optimisation problem.
        """
        self._model = model.new_copy()
        self._initial_state = initial_state
        self._solver = solver or model.default_solver
        self._parameter_values = (
            parameter_values.copy()
            if parameter_values
            else model.default_parameter_values
        )

    def add_cost(self, cost: PybammCost, weight: float = 1.0) -> None:
        """
        Add a cost to the problem with optional weighting.
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

            # Add hypers to pybop parameters
            if cost.metadata().parameters:
                for name, obj in cost.metadata().parameters.items():
                    pybop_parameters.add(
                        PybopParameter(name, initial_value=obj.default_value)
                    )

            # Posterior Logic
            if isinstance(cost, BaseLikelihood) and pybop_parameters.priors():
                self._use_posterior = True

            # Design Costs
            if isinstance(cost, DesignCost):
                self.cell_mass(pybamm_parameter_values)

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

    @staticmethod
    def cell_mass(parameter_values: pybamm.ParameterValues) -> None:
        """
        Calculate the total cell mass in kilograms.

        This method uses the provided parameter set to calculate the mass of different
        components of the cell, such as electrodes, separator, and current collectors,
        based on their densities, porosities, and thicknesses. It then calculates the
        total mass by summing the mass of each component and adds it as a parameter,
        `Cell mass [kg]` in the parameter_values dictionary.

        Parameters
        ----------
        parameter_values : dict
            A dictionary containing the parameter values necessary for the calculation.

        """
        params = parameter_values

        # Pre-calculate cross-sectional area
        cross_sectional_area = pybamm.Parameter(
            "Electrode height [m]"
        ) * pybamm.Parameter("Electrode width [m]")

        def electrode_mass_density(electrode_type):
            """Calculate mass density for positive or negative electrode."""
            prefix = f"{electrode_type} electrode"
            active_vol_frac = pybamm.Parameter(
                f"{prefix} active material volume fraction"
            )
            density = pybamm.Parameter(f"{prefix} active material density [kg.m-3]")
            porosity = pybamm.Parameter(f"{prefix} porosity")
            electrolyte_density = pybamm.Parameter("Electrolyte density [kg.m-3]")
            cb_density = pybamm.Parameter(f"{prefix} carbon-binder density [kg.m-3]")

            return (
                active_vol_frac * density
                + porosity * electrolyte_density
                + (1.0 - active_vol_frac - porosity) * cb_density
            )

        # Calculate all area densities
        area_densities = [
            # Electrodes
            pybamm.Parameter("Positive electrode thickness [m]")
            * electrode_mass_density("Positive"),
            pybamm.Parameter("Negative electrode thickness [m]")
            * electrode_mass_density("Negative"),
            # Separator
            pybamm.Parameter("Separator thickness [m]")
            * pybamm.Parameter("Separator density [kg.m-3]"),
            # Current collectors
            pybamm.Parameter("Positive current collector thickness [m]")
            * pybamm.Parameter("Positive current collector density [kg.m-3]"),
            pybamm.Parameter("Negative current collector thickness [m]")
            * pybamm.Parameter("Negative current collector density [kg.m-3]"),
        ]

        # Add cell mass to parameter_values
        params.update(
            {"Cell mass [kg]": cross_sectional_area * sum(area_densities)},
            check_already_exists=False,
        )
