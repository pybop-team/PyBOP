import pybamm

import pybop
from pybop import Parameter as PybopParameter
from pybop.builders.base import BaseBuilder
from pybop.builders.utils import cell_mass, set_formation_concentrations
from pybop.costs.pybamm import BaseLikelihood, DesignCost, PybammCost
from pybop.pipelines._pybamm_pipeline import PybammPipeline


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
        parameter_values: pybamm.ParameterValues | None = None,
        initial_state: float | str | None = None,
        experiment: pybamm.Experiment | None = None,
        solver: pybamm.BaseSolver | None = None,
        geometry: pybamm.Geometry | None = None,
        submesh_types: dict | None = None,
        var_pts: dict | None = None,
        spatial_methods: dict | None = None,
        discretisation_kwargs: dict | None = None,
        build_on_eval: bool = False,
    ) -> "Pybamm":
        """
        Adds a simulation for the optimisation problem.

        Parameters
        ----------
        model : pybamm.BaseModel
            The PyBaMM model to be used.
        parameter_values : pybamm.ParameterValues
            The parameters to be used in the model.
        initial_state : float | str
            The initial state of charge or voltage for the battery model. If float, it will be represented
            as SoC and must be in range 0 to 1. If str, it will be represented as voltage and needs to be in
            the format: "3.4 V".
        experiment : pybamm.Experiment
            The experiment to use.
        solver : pybamm.BaseSolver
            The solver to be used. If None, uses `pybop.RecommendedSolver`.
        geometry : pybamm.Geometry, optional
            The geometry upon which to solve the model.
        submesh_types : dict, optional
            A dictionary of the types of submesh to use on each subdomain.
        var_pts : dict, optional
            A dictionary of the number of points used by each spatial variable.
        spatial_methods : dict, optional
            A dictionary of the types of spatial method to use on each domain (e.g. pybamm.FiniteVolume).
        discretisation_kwargs : dict, optional
            Any keyword arguments to pass to the Discretisation class.
            See :class:`pybamm.Discretisation` for details.
        build_on_eval : bool, optional
            If True, the model will be rebuilt every evaluation. Otherwise, the need to rebuild will be
            determined automatically.
        """
        self._model = model.new_copy()
        self._parameter_values = (
            parameter_values.copy()
            if parameter_values
            else model.default_parameter_values
        )
        self._initial_state = initial_state
        self._experiment = experiment
        self._solver = solver
        self._geometry = geometry
        self._submesh_types = submesh_types
        self._var_pts = var_pts
        self._spatial_methods = spatial_methods
        self._discretisation_kwargs = discretisation_kwargs
        self._build_on_eval = build_on_eval

        return self

    def add_cost(self, cost: PybammCost, weight: float = 1.0) -> "Pybamm":
        """
        Adds a cost to the problem with optional weighting.
        """
        self._costs.append(cost)
        self._cost_weights.append(weight)

        return self

    def remove_costs(self) -> "Pybamm":
        """
        Removes all costs and corresponding weights from the problem.
        """
        self._costs = []
        self._cost_weights = []

        return self

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
        for cost in self._costs:
            cost.add_to_model(model, pybamm_parameter_values, self._dataset)
            cost_names.append(cost.metadata().variable_name)

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
                            name,
                            initial_value=obj.default_value,
                            prior=prior,
                            bounds=[0, obj.default_value * 20],
                        )
                    )

            # Design Costs
            if isinstance(cost, DesignCost):
                cell_mass(pybamm_parameter_values)
                if cost.set_formation_concentrations:
                    set_formation_concentrations(pybamm_parameter_values)

        # Construct the pipeline
        pipeline = PybammPipeline(
            model,
            cost_names=cost_names,
            pybop_parameters=pybop_parameters,
            parameter_values=pybamm_parameter_values,
            initial_state=self._initial_state,
            t_eval=time_params["t_eval"],
            t_interp=time_params["t_interp"],
            experiment=self._experiment,
            solver=self._solver,
            geometry=self._geometry,
            submesh_types=self._submesh_types,
            var_pts=self._var_pts,
            spatial_methods=self._spatial_methods,
            discretisation_kwargs=self._discretisation_kwargs,
            build_on_eval=self._build_on_eval,
        )

        return pybop.PybammProblem(
            pybamm_pipeline=pipeline,
            pybop_params=pybop_parameters,
            cost_names=cost_names,
            cost_weights=self._cost_weights,
            is_posterior=self.is_posterior,
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
            raise ValueError(
                "A dataset or an experiment must be provided before building"
            )

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
