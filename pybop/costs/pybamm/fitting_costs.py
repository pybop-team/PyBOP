import numpy as np
import pybamm

from pybop import Dataset
from pybop import Parameter as PybopParameter
from pybop.costs.pybamm.base_cost import (
    BaseLikelihood,
    PybammExpressionMetadata,
    PybammVariable,
)


class PybammErrorMeasure(PybammVariable):
    """
    A base class for error measure implementations within Pybamm.
    """

    def __init__(self, variable_name: str = None, data_name: str = None):
        super().__init__()
        self._variable_name = variable_name
        self._data_name = data_name

    def variable_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammExpressionMetadata:
        """
        Defines the variable name, expression, and any parameters that are
        needed to evaluate the cost function.
        """
        name = PybammErrorMeasure.make_unique_variable_name()
        self._check_state(dataset, model, name)
        data, var = self._construct_discrete_time_node(dataset, model, name)

        return PybammExpressionMetadata(
            variable_name=name,
            expression=self.expression(var=var, data=data),
            parameters={},
        )

    def expression(self, var, data):
        raise NotImplementedError


class SumSquaredError(PybammErrorMeasure):
    """
    Sum of squared error (SSE) cost function.

    Compute the sum of the squares of the differences between model predictions
    and target data, which serves as a measure of the total error between the
    predicted and observed values.
    """

    def __init__(self, variable_name: str, data_name: str):
        super().__init__(variable_name=variable_name, data_name=data_name)

    def expression(self, var, data):
        return pybamm.DiscreteTimeSum((var - data) ** 2)


class MeanAbsoluteError(PybammErrorMeasure):
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
        super().__init__(variable_name=variable_name, data_name=data_name)

    def expression(self, var, data):
        abs_diff = pybamm.AbsoluteValue(var - data)
        return pybamm.DiscreteTimeSum(abs_diff) / len(data.y)


class MeanSquaredError(PybammErrorMeasure):
    """
    Mean square error (MSE) cost function.

    Computes the mean square error between model predictions and the target
    data, providing a measure of the differences between predicted values and
    observed values.
    """

    def __init__(self, variable_name: str, data_name: str):
        super().__init__(variable_name=variable_name, data_name=data_name)

    def expression(self, var, data):
        return pybamm.DiscreteTimeSum((var - data) ** 2) / len(data.y)


class RootMeanSquaredError(PybammErrorMeasure):
    """
    Root mean square error (RMSE) cost function.

    Computes the root mean square error between model predictions and the target
    data, providing a measure of the differences between predicted values and
    observed values.
    """

    def __init__(self, variable_name: str, data_name: str):
        super().__init__(variable_name=variable_name, data_name=data_name)

    def expression(self, var, data):
        squared_error = (var - data) ** 2
        mean_squared_error = pybamm.DiscreteTimeSum(squared_error) / len(data.y)
        return pybamm.sqrt(mean_squared_error)


class Minkowski(PybammErrorMeasure):
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
        super().__init__(variable_name=variable_name, data_name=data_name)
        if p <= 0:
            raise ValueError(
                "The order of the Minkowski distance must be greater than 0."
            )
        elif not np.isfinite(p):
            raise ValueError(
                "For p = infinity, an implementation of the Chebyshev distance is required."
            )
        self._p = p

    def expression(self, var, data):
        abs_diff = pybamm.AbsoluteValue(var - data)
        return pybamm.DiscreteTimeSum(abs_diff**self._p) ** (1 / self._p)


class SumOfPower(PybammErrorMeasure):
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
        super().__init__(variable_name=variable_name, data_name=data_name)
        if p < 0:
            raise ValueError("The order of 'p' must be greater than 0.")
        elif not np.isfinite(p):
            raise ValueError("p = np.inf is not yet supported.")
        self._p = float(p)

    def expression(self, var, data):
        abs_diff = pybamm.AbsoluteValue(var - data)
        return pybamm.DiscreteTimeSum(abs_diff**self._p)


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
        sigma: float | PybopParameter | None = None,
    ):
        super().__init__()
        self._sigma = sigma
        self._variable_name = variable_name
        self._data_name = data_name
        self._log2pi = np.log(2 * np.pi)

    def variable_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammExpressionMetadata:
        """
        Defines the variable name, expression, and any parameters that are
        needed to evaluate the cost function.
        """
        name = NegativeGaussianLogLikelihood.make_unique_variable_name()
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


class ScaledCost(PybammVariable):
    """
    This class scales a BaseCost class by the number of observations.
    The scaling factor is given below:

    .. math::
       \\mathcal{\\hat{L(\theta)}} = \frac{1}{N} \\mathcal{L(\theta)}

    This class returns scaled numerical values with lower magnitude than the
    BaseCost, which can improve optimiser convergence in certain cases.
    """

    def __init__(self, cost: PybammVariable):
        super().__init__()
        self._cost = cost

    def variable_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammExpressionMetadata:
        """
        Defines the variable name, expression, and any parameters that are
        needed to evaluate the cost function.
        """
        name = ScaledCost.make_unique_variable_name()
        cost_metadata = self._cost.variable_expression(model, dataset)

        return PybammExpressionMetadata(
            variable_name=name,
            expression=cost_metadata.expression
            * pybamm.Scalar(1.0)
            / len(dataset[self._cost.data_name]),
            parameters=cost_metadata.parameters,
        )
