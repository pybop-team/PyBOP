import pybamm

from pybop import Dataset
from pybop import Parameter as PybopParameter
from pybop.costs.pybamm.output_variable import (
    PybammExpressionMetadata,
    PybammOutputVariable,
)


class UserCost(PybammOutputVariable):
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

    def symbolic_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammExpressionMetadata:
        """
        Returns the expression metadata for the cost function.
        """
        return self._metadata


def custom(
    name: str,
    expression: pybamm.Symbol,
    parameters: dict[str, pybamm.Parameter],
) -> PybammOutputVariable:
    """
    Creates a custom user-defined cost function for PyBaMM models.

    Parameters
    ----------
    name : str
        The name of the variable, must be unique within the variables of the model.
    expression : pybamm.Symbol
        The PyBaMM expression that defines the cost function.
    parameters : dict[str, pybamm.Parameter]
        A dictionary of parameters used in the cost function, where keys are parameter names and values are
        `pybop.Parameter` instances.

    Example
    -------
    >>> import pybop
    >>> import pybamm
    >>> builder = pybop.Pybamm()
    >>> builder.set_simulation(model)
    >>> builder.set_dataset(dataset)
    >>> builder.add_parameter(one_parameter)

    >>> # Create a custom cost
    >>> data = pybamm.DiscreteTimeData(
    >>>     dataset["Time [s]"], dataset["Voltage [V]"], "my_data"
    >>> )
    >>> custom_cost = pybop.costs.pybamm.custom(
    >>>     "MySumSquaredError",
    >>>     pybamm.DiscreteTimeSum((model.variables["Voltage [V]"] - data) ** 2),
    >>>     {},
    >>> )
    >>> builder.add_cost(custom_cost)
    >>> problem_custom = builder.build()
    """
    return UserCost(name, expression, parameters)
