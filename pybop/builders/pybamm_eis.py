from pybop import Parameter, PybammEISProblem, builders
from pybop._pybamm_eis_pipeline import PybammEISPipeline
from pybop.costs.pybamm_cost import PybammCost


class PybammEIS(builders.Pybamm):
    def __init__(self):
        super().__init__()
        self.domain = "Frequency [Hz]"

    def add_cost(self, cost: PybammCost, weight: float = 1.0) -> None:
        self._costs.append(cost)
        self._cost_names.append(cost.variable_name())
        self._cost_weights.append(weight)

    def add_parameter(self, parameter: Parameter) -> None:
        self._pybop_parameters.add(parameter)

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

        if self._pybamm_model is None:
            raise ValueError("A Pybamm model needs to be provided before building.")

        if self._costs is None:
            raise ValueError("A cost must be provided before building.")

        if self._dataset is None:
            raise ValueError("A dataset must be provided before building.")

        # Proceed to building the pipeline
        model = self._pybamm_model
        param = self._parameter_values
        pybop_parameters = self._pybop_parameters

        # Build pybamm if not already built
        if not model._built:  # noqa: SLF001
            model.build_model()

        # add costs
        for cost in self._costs:
            cost.add_to_model(model, param)

        # Construct the pipeline
        pipeline = PybammEISPipeline(
            model,
            param,
            pybop_parameters,
            self._solver,
            f_eval=self._dataset[self.domain],
        )

        # Build and initialise the pipeline
        pipeline.build()

        return PybammEISProblem(
            pybamm_pipeline=pipeline,
            pybop_params=self._pybop_parameters,
            cost_names=self._cost_names,
            cost_weights=self._cost_weights,
        )
