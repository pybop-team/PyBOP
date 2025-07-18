from collections.abc import Callable

import numpy as np
import pybamm

import pybop
from pybop import PybammEISProblem, builders
from pybop.costs.base_cost import CallableCost
from pybop.pipelines._pybamm_eis_pipeline import PybammEISPipeline


class PybammEIS(builders.BaseBuilder):
    def __init__(self):
        super().__init__()
        self._model = None
        self._geometry = None
        self._parameter_values = None
        self._submesh_types = None
        self._var_pts = None
        self._spatial_methods = None
        self._solver = None
        self._initial_state = None
        self._build_on_eval = None
        self._pipeline = None
        self._rebuild_parameters = None
        self.domain = "Frequency [Hz]"
        self._costs: list[CallableCost] = []
        self._cost_weights: list[float] = []

    def set_simulation(
        self,
        model: pybamm.BaseModel,
        geometry: pybamm.Geometry | None = None,
        parameter_values: pybamm.ParameterValues | None = None,
        submesh_types: dict | None = None,
        var_pts: dict | None = None,
        spatial_methods: dict | None = None,
        initial_state: dict | None = None,
        build_on_eval: bool = False,
    ) -> None:
        """
        Adds a simulation for the optimisation problem.

        Parameters
        ----------
        model : pybamm.BaseModel
            The PyBaMM model to be used.
        geometry: pybamm.Geometry (optional)
            The geometry upon which to solve the model.
        parameter_values : pybamm.ParameterValues (optional)
            Parameters and their corresponding numerical values.
        submesh_types : dict (optional)
            A dictionary of the types of submesh to use on each subdomain.
        var_pts : dict (optional)
            A dictionary of the number of points used by each spatial variable.
        spatial_methods : dict (optional)
            A dictionary of the types of spatial method to use on each.
            domain (e.g. pybamm.FiniteVolume)
        initial_state: dict (optional)
            A valid initial state, e.g. the initial state of charge or open-circuit voltage.
        build_on_eval : bool
            Boolean to determine if the model will be rebuilt every evaluation. If `initial_state` is provided,
            the model will be rebuilt every evaluation unless `build_on_eval` is `False`, in which case the model
            is built with the parameter values from construction only.
        """
        self._model = model.new_copy()
        self._geometry = geometry
        self._parameter_values = (
            parameter_values.copy()
            if parameter_values
            else model.default_parameter_values
        )
        self._submesh_types = submesh_types
        self._var_pts = var_pts
        self._spatial_methods = spatial_methods
        self._solver = pybamm.CasadiSolver()
        self._initial_state = initial_state
        self._build_on_eval = build_on_eval

    def add_cost(self, cost: Callable | CallableCost, weight: float = 1.0) -> None:
        """Adds a cost to the problem."""
        if not isinstance(cost, CallableCost):
            if not isinstance(cost, Callable):
                raise TypeError(
                    "cost must be a callable or an instance of CallableCost"
                )
            cost = pybop.costs.CallableError(cost)
        if cost.weighting is None or cost.weighting == "equal":
            cost.weighting = np.array(1.0)
        elif cost.weighting == "domain":
            self._set_cost_domain_weighting(cost)
        else:
            raise ValueError(
                "cost.weighting must be 'equal', 'domain', or a custom numpy array"
                f", got {cost.weighting}"
            )

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
            f_eval=self._dataset[self.domain],
            geometry=self._geometry,
            parameter_values=param,
            submesh_types=self._submesh_types,
            var_pts=self._var_pts,
            spatial_methods=self._spatial_methods,
            solver=self._solver,
            pybop_parameters=pybop_parameters,
            initial_state=self._initial_state,
            build_on_eval=self._build_on_eval,
        )

        # Build and initialise the pipeline
        pipeline.build()

        return PybammEISProblem(
            eis_pipeline=pipeline,
            pybop_params=pybop_parameters,
            costs=self._costs,
            cost_weights=self._cost_weights,
            fitting_data=self._dataset["Impedance"],
        )
