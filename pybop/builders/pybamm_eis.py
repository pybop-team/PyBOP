from collections.abc import Callable

import pybamm

import pybop
from pybop import PybammEISProblem, builders
from pybop.costs.base_cost import CallableCost
from pybop.pipelines._pybamm_eis_pipeline import PybammEISPipeline


class PybammEIS(builders.BaseBuilder):
    def __init__(self):
        super().__init__()
        self.domain = "Frequency [Hz]"
        self._costs: list[CallableCost] = []
        self._cost_weights: list[float] = []

    def set_simulation(
        self,
        model: pybamm.BaseModel,
        parameter_values: pybamm.ParameterValues | None = None,
        initial_state: float | str | None = None,
        geometry: pybamm.Geometry | None = None,
        submesh_types: dict | None = None,
        var_pts: dict | None = None,
        spatial_methods: dict | None = None,
        discretisation_kwargs: dict | None = None,
        build_on_eval: bool | None = None,
    ) -> None:
        """
        Adds a simulation for the optimisation problem.

        Parameters
        ----------
        model : pybamm.BaseModel
            The PyBaMM model to be used.
        parameter_values : pybamm.ParameterValues
            The parameters to be used in the model.
        initial_state: float | str
            The initial state of charge or voltage for the battery model. If float, it will be represented
            as SoC and must be in range 0 to 1. If str, it will be represented as voltage and needs to be in
            the format: "3.4 V".
        geometry : pybamm.Geometry, optional
            The geometry upon which to solve the model.
        submesh_types : dict, optional
            A dictionary of the types of submesh to use on each subdomain.
        var_pts : dict, optional
            A dictionary of the number of points used by each spatial variable.
        spatial_methods : dict, optional
            A dictionary of the types of spatial method to use on each domain (e.g. pybamm.FiniteVolume).
        discretisation_kwargs : dict (optional)
            Any keyword arguments to pass to the Discretisation class.
            See :class:`pybamm.Discretisation` for details.
        build_on_eval : bool
            Boolean to determine if the model will be rebuilt every evaluation. If `initial_state` is provided,
            the model will be rebuilt every evaluation unless `build_on_eval` is `False`, in which case the model
            is built with the parameter values from construction only.
        """
        self._model = model.new_copy()
        self._parameter_values = (
            parameter_values.copy()
            if parameter_values
            else model.default_parameter_values
        )
        self._initial_state = initial_state
        self._solver = pybamm.CasadiSolver()
        self._geometry = geometry
        self._submesh_types = submesh_types
        self._var_pts = var_pts
        self._spatial_methods = spatial_methods
        self._discretisation_kwargs = discretisation_kwargs
        self._build_on_eval = build_on_eval

    def add_cost(
        self, cost: Callable | CallableCost, weight: float = 1.0
    ) -> "PybammEIS":
        """Adds a cost to the problem."""
        if not isinstance(cost, CallableCost):
            if not isinstance(cost, Callable):
                raise TypeError(
                    "cost must be a callable or an instance of CallableCost"
                )
            cost = pybop.costs.CallableError(cost)

        # Set the time-series weighting
        cost.weighting = builders.create_weighting(
            cost.weighting, self._dataset, self.domain
        )

        self._costs.append(cost)
        self._cost_weights.append(weight)

        return self

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

        if not self._costs:
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
            pybop_parameters=pybop_parameters,
            parameter_values=param,
            initial_state=self._initial_state,
            solver=self._solver,
            geometry=self._geometry,
            submesh_types=self._submesh_types,
            var_pts=self._var_pts,
            spatial_methods=self._spatial_methods,
            discretisation_kwargs=self._discretisation_kwargs,
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
