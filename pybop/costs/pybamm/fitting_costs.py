from typing import Optional, Union

import numpy as np
import pybamm

from pybop import Dataset
from pybop import Parameter as PybopParameter
from pybop.costs.pybamm.base_cost import (
    BaseCost,
    BaseLikelihood,
    PybammExpressionMetadata,
)


class SumSquaredError(BaseCost):
    """
    A SumSquaredError cost implementation within Pybamm.
    """

    def __init__(
        self,
        variable_name: str,
        data_name: str,
        sigma: Optional[Union[float, PybopParameter]] = None,
    ):
        super().__init__()
        self._sigma = sigma
        self._variable_name = variable_name
        self._data_name = data_name

    def variable_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Optional[Dataset] = None,
    ) -> PybammExpressionMetadata:
        # Check args
        name = SumSquaredError.make_unique_cost_name()
        self._check_state(dataset, model, name)
        data, var = self._construct_discrete_time_node(dataset, model, name)
        parameters = {}
        sigma = self._get_sigma_parameter(name, parameters)
        sum_expr = (var - data) ** 2 / sigma**2

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
        dataset: Optional[Dataset] = None,
    ) -> PybammExpressionMetadata:
        # Check args
        name = SumSquaredError.make_unique_cost_name()
        self._check_state(dataset, model, name)
        data, var = self._construct_discrete_time_node(dataset, model, name)

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
        dataset: Optional[Dataset] = None,
    ) -> PybammExpressionMetadata:
        # Check args
        name = MeanSquaredError.make_unique_cost_name()
        self._check_state(dataset, model, name)
        data, var = self._construct_discrete_time_node(dataset, model, name)

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
        dataset: Optional[Dataset] = None,
    ) -> PybammExpressionMetadata:
        # Check args
        name = RootMeanSquaredError.make_unique_cost_name()
        self._check_state(dataset, model, name)
        data, var = self._construct_discrete_time_node(dataset, model, name)

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
        dataset: Optional[Dataset] = None,
    ) -> PybammExpressionMetadata:
        # Check args
        name = Minkowski.make_unique_cost_name()
        self._check_state(dataset, model, name)
        data, var = self._construct_discrete_time_node(dataset, model, name)

        # Create Expression
        diff = var - data
        p = pybamm.Scalar(self._p)
        abs_diff = pybamm.AbsoluteValue(diff)
        powered_diff = pybamm.Power(abs_diff, p)
        sum_powered = pybamm.DiscreteTimeSum(powered_diff)
        expression = pybamm.Power(sum_powered, 1 / p)

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
        dataset: Optional[Dataset] = None,
    ) -> PybammExpressionMetadata:
        # Check args
        name = SumOfPower.make_unique_cost_name()
        self._check_state(dataset, model, name)
        data, var = self._construct_discrete_time_node(dataset, model, name)

        # Create Expression
        diff = var - data
        abs_diff = pybamm.AbsoluteValue(diff)
        sum_expr = abs_diff**self._p

        return PybammExpressionMetadata(
            variable_name=name,
            expression=pybamm.DiscreteTimeSum(sum_expr),
            parameters={},
        )


class NegativeGaussianLogLikelihood(BaseLikelihood):
    """
    This class represents a Gaussian log-likelihood, which computes the log-likelihood under
    the assumption that measurement noise on the target data follows a Gaussian distribution.

    This class estimates the standard deviation of the Gaussian distribution alongside the
    parameters of the model.

    """

    def __init__(
        self,
        variable_name: str,
        data_name: str,
        sigma: Optional[Union[float, PybopParameter]] = None,
    ):
        super().__init__()
        self._sigma = sigma
        self._variable_name = variable_name
        self._data_name = data_name
        self._log2pi = np.log(2 * np.pi)

    def variable_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Optional[Dataset] = None,
    ) -> PybammExpressionMetadata:
        # Check args
        name = NegativeGaussianLogLikelihood.make_unique_cost_name()
        self._check_state(dataset, model, name)
        data, var = self._construct_discrete_time_node(dataset, model, name)

        parameters = {}
        sigma = self._get_sigma_parameter(name, parameters)
        sum_expr = -1 * (
            -0.5 * self._log2pi
            - pybamm.log(sigma)
            - (var - data) ** 2 / (2.0 * sigma**2.0)
        )

        return PybammExpressionMetadata(
            variable_name=name,
            expression=pybamm.DiscreteTimeSum(sum_expr),
            parameters=parameters,
        )


class ScaledCost(BaseCost):
    """
    This class scales a BaseCost class by the number of observations.
    The scaling factor is given below:

    .. math::
       \\mathcal{\\hat{L(\theta)}} = \frac{1}{N} \\mathcal{L(\theta)}

    This class returns scaled numerical values with lower magnitude than the
    BaseCost, which can improve optimiser convergence in certain cases.
    """

    def __init__(
        self,
        loglikelihood: BaseCost,
    ):
        super().__init__()
        self._loglikelihood = loglikelihood

    def variable_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Optional[Dataset] = None,
    ) -> PybammExpressionMetadata:
        # Check args
        name = NegativeGaussianLogLikelihood.make_unique_cost_name()
        loglikelihood_metadata = self._loglikelihood.variable_expression(model, dataset)

        return PybammExpressionMetadata(
            variable_name=name,
            expression=loglikelihood_metadata.expression
            * pybamm.Scalar(1.0)
            / len(dataset[self._data_name]),
            parameters=loglikelihood_metadata.parameters,
        )
