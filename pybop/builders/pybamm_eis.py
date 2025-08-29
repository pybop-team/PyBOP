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
        if parameter_values is None:
            parameter_values = model.default_parameter_values
        elif isinstance(parameter_values, pybamm.ParameterValues):
            parameter_values = parameter_values.copy()
        else:
            raise TypeError(
                "parameter_values must be a pybamm.ParameterValues instance or None"
            )
        self._parameter_values = parameter_values
        self._solver = pybamm.CasadiSolver()

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
            self._dataset[self.domain],
            param,
            pybop_parameters,
            self._solver,
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
