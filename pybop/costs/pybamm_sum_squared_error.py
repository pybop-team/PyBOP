from typing import Optional

import pybamm

import pybop


class PybammSumSquaredError(pybop.PybammCost):
    """
    A SumSquaredError cost implementation within Pybamm.
    """

    def __init__(
        self, variable_name: str, data_name: str, sigma: Optional[float] = None
    ):
        super().__init__()
        self._sigma = sigma
        self._variable_name = variable_name
        self._data_name = data_name

    @staticmethod
    def variable_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Optional[pybop.Dataset] = None,
    ) -> pybop.PybammExpressionMetadata:
        # dataset must be provided and contain the data
        if dataset is None:
            raise ValueError("Dataset must be provided for PybammSumSquaredError.")
        if self._data_name not in dataset:
            raise ValueError(
                f"Dataset must contain {self._data_name} for PybammSumSquaredError."
            )
        # model must contain the variable
        if self._variable_name not in model.variables:
            raise ValueError(
                f"Model must contain {self._variable_name} for PybammSumSquaredError."
            )
        times = dataset["Time [s]"]
        values = dataset[self._data_name]
        name = PybammSumSquaredError.make_unique_cost_name()
        data = pybamm.DiscreteTimeData(times, values, f"{name}_data")
        var = model.variables[self._variable_name]
        parameters = {}
        if self._sigma is None:
            sigma_name = f"{name}_sigma"
            sigma = pybamm.Parameter(sigma_name)
            sum_expr = (var - data) ** 2 / sigma**2
            parameters[sigma_name] = pybop.PybammParameterMetadata(
                parameter=sigma, default_value=1.0
            )
        else:
            sum_expr = (var - data) ** 2 / self._sigma**2

        return pybop.PybammExpressionMetadata(
            variable_name=name,
            expression=pybamm.DiscreteTimeSum(sum_expr),
            parameters=parameters,
        )
