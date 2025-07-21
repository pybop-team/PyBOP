import pybamm

from pybop import Dataset
from pybop import Parameter as PybopParameter
from pybop.costs.pybamm.base_cost import (
    PybammCost,
    PybammExpressionMetadata,
)


class UserCost(PybammCost):
    """
    A user-defined cost function for PyBaMM models.
    This class allows users to define custom cost functions
    using PyBaMM expressions and parameters.
    """

    def __init__(
        self,
        name: str,
        expression: pybamm.Symbol,
        parameters: dict[str, PybopParameter],
    ):
        super().__init__()
        self._metadata = PybammExpressionMetadata(name, expression, parameters)
        self._variable_name = name

    def variable_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammExpressionMetadata:
        """
        Returns the expression metadata for the cost function.
        """
        return self._metadata


def custom(
    variable_name: str,
    expression: pybamm.Symbol,
    parameters: dict[str, pybamm.Parameter],
) -> PybammCost:
    return UserCost(variable_name, expression, parameters)
