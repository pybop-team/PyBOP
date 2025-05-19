from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

import numpy as np
import pybamm

import pybop


@dataclass
class PybammExpressionMetadata:
    """
    Metadata for a PyBaMM cost function. This includes the variable name, expression,
    and any parameters that are needed to evaluate the cost function.
    """

    variable_name: str
    expression: pybamm.Symbol
    parameters: dict[str, pybamm.Parameter]


@dataclass
class PybammParameterMetadata:
    """
    Metadata for a PyBaMM parameter. This includes the
    pybamm parameter and the default value to use if the parameter is not
    set explicitly.
    """

    parameter: pybamm.Parameter
    default_value: float


class BaseCost:
    def __init__(self):
        self._metadata = None
        self._data_name = None
        self._variable_name = None
        self._sigma = None

    def metadata(self) -> PybammExpressionMetadata:
        """
        Returns the metadata for the cost function. This includes the variable name,
        expression, and any parameters that are needed to evaluate the cost function.
        """
        if self._metadata is None:
            raise ValueError("Cost function has not been added to model yet.")
        return self._metadata

    @classmethod
    def make_unique_cost_name(cls) -> str:
        """
        Make a unique name for the cost function variable using the name of the class
        and a UUID. This is used to avoid name collisions in the pybamm model.
        """
        return f"{cls.__class__.__name__}_{uuid4()}"

    def variable_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Optional[pybop.Dataset] = None,
    ) -> PybammExpressionMetadata:
        """
        Defines the variable expression for the cost function, returning a
        PybammExpressionMetadata object. This should be implemented in the
        subclass. The metadata object contains the variable name, expression,
        and any parameters that are needed to evaluate the cost function.
        """
        raise NotImplementedError()

    def add_to_model(
        self,
        model: pybamm.BaseModel,
        param: pybamm.ParameterValues,
        dataset: Optional[pybop.Dataset] = None,
    ):
        # if dataset is provided, must contain time data
        if dataset is not None and "Time [s]" not in dataset:
            raise ValueError("Dataset must contain time data for PybammCost.")
        self._metadata = self.variable_expression(model, dataset)
        model.variables[self._metadata.variable_name] = self._metadata.expression
        for parameter_name, parameter_metadata in self._metadata.parameters.items():
            param.update(
                {parameter_name: parameter_metadata.default_value},
                check_already_exists=False,
            )

    def _check_state(self, dataset, model, name) -> None:
        # dataset must be provided and contain the data
        if dataset is None:
            raise ValueError(f"Dataset must be provided for {name}.")
        if self._data_name not in dataset:
            raise ValueError(f"Dataset must contain {self._data_name} for {name}.")
        # model must contain the variable
        if self._variable_name not in model.variables:
            raise ValueError(f"Model must contain {self._variable_name} for {name}.")


class SumSquaredError(BaseCost):
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

    def variable_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Optional[pybop.Dataset] = None,
    ) -> PybammExpressionMetadata:
        # Check args
        name = SumSquaredError.make_unique_cost_name()
        self._check_state(dataset, model, name)

        # Construct cost
        times = dataset["Time [s]"]
        values = dataset[self._data_name]
        data = pybamm.DiscreteTimeData(times, values, f"{name}_data")
        var = model.variables[self._variable_name]

        # Create Expression
        parameters = {}
        if self._sigma is None:
            sigma_name = f"sigma_{name}"
            sigma = pybamm.Parameter(sigma_name)
            sum_expr = (var - data) ** 2 / sigma**2
            parameters[sigma_name] = PybammParameterMetadata(
                parameter=sigma, default_value=1.0
            )
        else:
            sum_expr = (var - data) ** 2 / self._sigma**2

        return PybammExpressionMetadata(
            variable_name=name,
            expression=pybamm.DiscreteTimeSum(sum_expr),
            parameters=parameters,
        )


class MeanAbsoluteError(BaseCost):
    """
    Mean absolute error (MAE) cost function.

    Computes the mean absolute error (MAE) between model predictions
    and target data. The MAE is a measure of the average magnitude
    of errors in a set of predictions, without considering their direction.

    Parameters
    ----------
    weighting : np.ndarray, optional
        The type of weighting array to use when taking the sum or mean of the error
        measure. Options: "equal"(default), "domain", or a custom numpy array.
    """

    def __init__(self, variable_name: str, data_name: str):
        super().__init__()
        self._variable_name = variable_name
        self._data_name = data_name

    def variable_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Optional[pybop.Dataset] = None,
    ) -> PybammExpressionMetadata:
        # Check args
        name = SumSquaredError.make_unique_cost_name()
        self._check_state(dataset, model, name)

        # Construct cost
        times = dataset["Time [s]"]
        values = dataset[self._data_name]
        name = SumSquaredError.make_unique_cost_name()
        data = pybamm.DiscreteTimeData(times, values, f"{name}_data")
        var = model.variables[self._variable_name]

        # Create Expression
        sum_expr = pybamm.AbsoluteValue(var - data) / len(data.y)

        return PybammExpressionMetadata(
            variable_name=name,
            expression=pybamm.DiscreteTimeSum(sum_expr),
            parameters={},
        )


class MeanSquaredError(BaseCost):
    """
    Mean square error (MSE) cost function.

    Computes the mean square error between model predictions and the target
    data, providing a measure of the differences between predicted values and
    observed values.
    """

    def __init__(self, variable_name: str, data_name: str):
        super().__init__()
        self._variable_name = variable_name
        self._data_name = data_name

    def variable_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Optional[pybop.Dataset] = None,
    ) -> PybammExpressionMetadata:
        # Check args
        name = MeanSquaredError.make_unique_cost_name()
        self._check_state(dataset, model, name)

        # Construct cost
        times = dataset["Time [s]"]
        values = dataset[self._data_name]
        data = pybamm.DiscreteTimeData(times, values, f"{name}_data")
        var = model.variables[self._variable_name]

        # Create Expression
        squared_error = (var - data) ** 2
        sum_expr = squared_error / len(data.y)

        return PybammExpressionMetadata(
            variable_name=name,
            expression=pybamm.DiscreteTimeSum(sum_expr),
            parameters={},
        )


class RootMeanSquaredError(BaseCost):
    """
    Root mean square error (RMSE) cost function.

    Computes the root mean square error between model predictions and the target
    data, providing a measure of the differences between predicted values and
    observed values.
    """

    def __init__(self, variable_name: str, data_name: str):
        super().__init__()
        self._variable_name = variable_name
        self._data_name = data_name

    def variable_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Optional[pybop.Dataset] = None,
    ) -> PybammExpressionMetadata:
        # Check args
        name = RootMeanSquaredError.make_unique_cost_name()
        self._check_state(dataset, model, name)

        # Construct cost
        times = dataset["Time [s]"]
        values = dataset[self._data_name]
        data = pybamm.DiscreteTimeData(times, values, f"{name}_data")
        var = model.variables[self._variable_name]

        # Create Expression
        squared_error = (var - data) ** 2
        mean_squared_error = squared_error / len(data.y)
        sum_expr = pybamm.sqrt(mean_squared_error)

        return PybammExpressionMetadata(
            variable_name=name,
            expression=pybamm.DiscreteTimeSum(sum_expr),
            parameters={},
        )


class Minkowski(BaseCost):
    """
    The Minkowski distance is a generalisation of several distance metrics,
    including the Euclidean and Manhattan distances. It is defined as:

    .. math::
        L_p(x, y) = ( \\sum_i |x_i - y_i|^p )^(1/p)

    where p > 0 is the order of the Minkowski distance.

    Special cases:
    * p = 1: Manhattan distance
    * p = 2: Euclidean distance
    """

    def __init__(self, variable_name: str, data_name: str, p: float = 2.0):
        super().__init__()
        if p <= 0:
            raise ValueError(
                "The order of the Minkowski distance must be greater than 0."
            )
        elif not np.isfinite(p):
            raise ValueError(
                "For p = infinity, an implementation of the Chebyshev distance is required."
            )
        self._variable_name = variable_name
        self._data_name = data_name
        self._p = p

    def variable_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Optional[pybop.Dataset] = None,
    ) -> PybammExpressionMetadata:
        # Check args
        name = Minkowski.make_unique_cost_name()
        self._check_state(dataset, model, name)

        # Construct cost
        times = dataset["Time [s]"]
        values = dataset[self._data_name]
        data = pybamm.DiscreteTimeData(times, values, f"{name}_data")
        var = model.variables[self._variable_name]

        # Create Expression
        diff = var - data
        abs_diff = pybamm.AbsoluteValue(diff)
        powered_diff = abs_diff**self._p
        sum_powered = pybamm.DiscreteTimeSum(powered_diff)
        expression = sum_powered ** (1 / self._p)

        return PybammExpressionMetadata(
            variable_name=name,
            expression=expression,
            parameters={},
        )


class SumOfPower(BaseCost):
    """
    The Sum of Power [1] is a generalised cost function based on the p-th power
    of absolute differences between two vectors. It is defined as:

    .. math::
        C_p(x, y) = \\sum_i |x_i - y_i|^p

    where p ≥ 0 is the power order.

    This class implements the Sum of Power as a cost function for
    optimisation problems, allowing for flexible power-based optimisation
    across various problem domains.

    Special cases:

    * p = 1: Sum of Absolute Differences
    * p = 2: Sum of Squared Differences
    * p → ∞: Maximum Absolute Difference

    Note that this is not normalised, unlike distance metrics. To get a
    distance metric, you would need to take the p-th root of the result.

    [1]: https://mathworld.wolfram.com/PowerSum.html
    """

    def __init__(self, variable_name: str, data_name: str, p: float = 2.0):
        super().__init__()
        if p < 0:
            raise ValueError("The order of 'p' must be greater than 0.")
        elif not np.isfinite(p):
            raise ValueError("p = np.inf is not yet supported.")
        self._variable_name = variable_name
        self._data_name = data_name
        self._p = float(p)

    def variable_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Optional[pybop.Dataset] = None,
    ) -> PybammExpressionMetadata:
        # Check args
        name = SumOfPower.make_unique_cost_name()
        self._check_state(dataset, model, name)

        # Construct cost
        times = dataset["Time [s]"]
        values = dataset[self._data_name]
        data = pybamm.DiscreteTimeData(times, values, f"{name}_data")
        var = model.variables[self._variable_name]

        # Create Expression
        diff = var - data
        abs_diff = pybamm.AbsoluteValue(diff)
        sum_expr = abs_diff**self._p

        return PybammExpressionMetadata(
            variable_name=name,
            expression=pybamm.DiscreteTimeSum(sum_expr),
            parameters={},
        )
