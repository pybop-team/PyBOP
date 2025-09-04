import numpy as np
import pybamm

from pybop import Dataset
from pybop import Parameter as PybopParameter
from pybop.costs.pybamm.base_cost import (
    BaseLikelihood,
    PybammExpressionMetadata,
    PybammVariable,
)


class SumSquaredError(PybammVariable):
    """
    A SumSquaredError cost implementation within Pybamm.
    """

    def __init__(
        self,
        variable_name: str,
        data_name: str,
    ):
        super().__init__()
        self._variable_name = variable_name
        self._data_name = data_name

    def symbolic_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammExpressionMetadata:
        # Check args
        name = SumSquaredError.make_unique_cost_name()
        self._check_state(dataset, model, name)
        data, var = self._construct_discrete_time_node(dataset, model, name)
        parameters = {}
        sum_expr = (var - data) ** 2

        return PybammExpressionMetadata(
            variable_name=name,
            expression=pybamm.DiscreteTimeSum(sum_expr),
            parameters=parameters,
        )

    def __name__(self):
        return "Sum Squared Error"


class MeanAbsoluteError(PybammVariable):
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

    def symbolic_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammExpressionMetadata:
        # Check args
        name = MeanAbsoluteError.make_unique_cost_name()
        self._check_state(dataset, model, name)
        data, var = self._construct_discrete_time_node(dataset, model, name)

        # Create Expression
        sum_expr = pybamm.AbsoluteValue(var - data) / len(data.y)

        return PybammExpressionMetadata(
            variable_name=name,
            expression=pybamm.DiscreteTimeSum(sum_expr),
            parameters={},
        )

    def __name__(self):
        return "Mean Absolute Error"


class MeanSquaredError(PybammVariable):
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

    def symbolic_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
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

    def __name__(self):
        return "Mean Squared Error"


class RootMeanSquaredError(PybammVariable):
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

    def symbolic_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
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

    def __name__(self):
        return "Root Mean Squared Error"


class Minkowski(PybammVariable):
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

    def symbolic_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammExpressionMetadata:
        # Check args
        name = Minkowski.make_unique_cost_name()
        self._check_state(dataset, model, name)
        data, var = self._construct_discrete_time_node(dataset, model, name)

        # Create Expression
        diff = var - data
        abs_diff = pybamm.AbsoluteValue(diff)
        expression = pybamm.DiscreteTimeSum(abs_diff**self._p) ** (1 / self._p)

        return PybammExpressionMetadata(
            variable_name=name,
            expression=expression,
            parameters={},
        )

    def __name__(self):
        return f"Minkowski distance (p = {self.p})"


class SumOfPower(PybammVariable):
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

    def symbolic_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
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

    def __name__(self):
        return f"Sum of Power (p = {self._p})"


class NegativeGaussianLogLikelihood(BaseLikelihood):
    """
    Negative Gaussian log-likelihood with Gaussian noise.

    This class computes the negative log-likelihood under the assumption that the
    measurement noise on the target data follows a Gaussian (normal) distribution.

    The likelihood function assumes:
    - Target values y are normally distributed around predicted values μ
    - Constant variance σ² across all observations (homoscedastic noise)

    The negative log-likelihood is computed as:
    -ℓ(θ, σ²) = n/2 * log(2πσ²) + 1/(2σ²) * Σ(yᵢ - μᵢ(θ))²

    where n is the number of observations, θ represents model parameters, and
    μᵢ(θ) is the predicted value for observation i.

    If a `sigma` argument is not provided, this implementation estimates
    both the model parameters θ and the noise standard deviation σ
    simultaneously during optimisation.

    Reference:
        Pawitan, Y. "In all likelihood: Statistical modelling and inference
        using likelihood"
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

    def symbolic_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammExpressionMetadata:
        # Check args
        name = NegativeGaussianLogLikelihood.make_unique_cost_name()
        self._check_state(dataset, model, name)
        data, var = self._construct_discrete_time_node(dataset, model, name)

        parameters = {}
        sigma = self._get_sigma_parameter(name, parameters)
        sum_expr = pybamm.DiscreteTimeSum((data - var) ** 2.0)

        nll = (
            data.size * self._log2pi / 2.0
            + data.size * pybamm.log(sigma)
            + sum_expr / (2.0 * sigma**2.0)
        )

        return PybammExpressionMetadata(
            variable_name=name,
            expression=nll,
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

    def __init__(
        self,
        cost: PybammVariable,
    ):
        super().__init__()
        self._cost = cost

    def symbolic_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammExpressionMetadata:
        # Check args
        name = ScaledCost.make_unique_cost_name()
        cost_metadata = self._cost.symbolic_expression(model, dataset)

        return PybammExpressionMetadata(
            variable_name=name,
            expression=cost_metadata.expression
            * pybamm.Scalar(1.0)
            / len(dataset[self._cost.data_name]),
            parameters=cost_metadata.parameters,
        )

    def __name__(self):
        return f"Scaled {self._cost.__name__()}"
