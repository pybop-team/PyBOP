import numpy as np
import pybamm

from pybop import Dataset
from pybop import Parameter as PybopParameter
from pybop.costs.pybamm.base_cost import (
    PybammParameterMetadata,
    PybammVariable,
    PybammVariableMetadata,
)


class PybammFittingCost(PybammVariable):
    """
    A base class for error measure implementations within Pybamm.
    """

    def __init__(self, signal: str | None = None):
        super().__init__()
        self._signal = signal

    def _check_state(self, dataset, model, cost_name) -> None:
        """
        Check that the variable on which the new variable depends exist in both the
        dataset and the model.
        """
        if dataset is None:
            raise ValueError(f"A dataset must be provided for {cost_name}.")
        if self._signal not in dataset:
            raise ValueError(f"Dataset must contain {self._signal} for {cost_name}.")
        if self._signal not in model.variables:
            raise ValueError(f"Model must contain {self._signal} for {cost_name}.")

    def _construct_discrete_time_node(self, dataset, model, cost_name):
        """
        Constructs the pybamm.DiscreteTimeData node in the expression tree and returns

        """
        self._check_state(dataset, model, cost_name)

        times = dataset["Time [s]"]
        values = dataset[self._signal]
        data = pybamm.DiscreteTimeData(times, values, f"{self._signal} (data)")
        var = model.variables[self._signal]
        return data, var

    @property
    def signal(self) -> str | None:
        return self._signal


class SumSquaredError(PybammFittingCost):
    """
    Sum of squared error (SSE) cost function.

    Compute the sum of the squares of the differences between model predictions
    and target data, which serves as a measure of the total error between the
    predicted and observed values.
    """

    def symbolic_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammVariableMetadata:
        """Construct the variable metadata."""
        name = SumSquaredError.make_unique_cost_name()

        data, var = self._construct_discrete_time_node(dataset, model, name)
        sum_expr = (var - data) ** 2

        return PybammVariableMetadata(
            variable_name=name,
            expression=pybamm.DiscreteTimeSum(sum_expr),
            parameters={},
        )

    def __name__(self):
        return "Sum Squared Error"


class MeanAbsoluteError(PybammFittingCost):
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

    def symbolic_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammVariableMetadata:
        """Construct the variable metadata."""
        name = MeanAbsoluteError.make_unique_cost_name()
        data, var = self._construct_discrete_time_node(dataset, model, name)

        sum_expr = pybamm.AbsoluteValue(var - data) / len(data.y)

        return PybammVariableMetadata(
            variable_name=name,
            expression=pybamm.DiscreteTimeSum(sum_expr),
            parameters={},
        )

    def __name__(self):
        return "Mean Absolute Error"


class MeanSquaredError(PybammFittingCost):
    """
    Mean square error (MSE) cost function.

    Computes the mean square error between model predictions and the target
    data, providing a measure of the differences between predicted values and
    observed values.
    """

    def symbolic_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammVariableMetadata:
        """Construct the variable metadata."""
        name = MeanSquaredError.make_unique_cost_name()

        data, var = self._construct_discrete_time_node(dataset, model, name)
        squared_error = (var - data) ** 2
        sum_expr = squared_error / len(data.y)

        return PybammVariableMetadata(
            variable_name=name,
            expression=pybamm.DiscreteTimeSum(sum_expr),
            parameters={},
        )

    def __name__(self):
        return "Mean Squared Error"


class RootMeanSquaredError(PybammFittingCost):
    """
    Root mean square error (RMSE) cost function.

    Computes the root mean square error between model predictions and the target
    data, providing a measure of the differences between predicted values and
    observed values.
    """

    def symbolic_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammVariableMetadata:
        """Construct the variable metadata."""
        name = RootMeanSquaredError.make_unique_cost_name()

        data, var = self._construct_discrete_time_node(dataset, model, name)
        squared_error = (var - data) ** 2
        mean_squared_error = squared_error / len(data.y)
        sum_expr = pybamm.sqrt(mean_squared_error)

        return PybammVariableMetadata(
            variable_name=name,
            expression=pybamm.DiscreteTimeSum(sum_expr),
            parameters={},
        )

    def __name__(self):
        return "Root Mean Squared Error"


class Minkowski(PybammFittingCost):
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

    def __init__(self, signal: str, p: float = 2.0):
        super().__init__(signal=signal)
        if p <= 0:
            raise ValueError(
                "The order of the Minkowski distance must be greater than 0."
            )
        elif not np.isfinite(p):
            raise ValueError(
                "For p = infinity, an implementation of the Chebyshev distance is required."
            )
        self._p = p

    def symbolic_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammVariableMetadata:
        """Construct the variable metadata."""
        name = Minkowski.make_unique_cost_name()

        data, var = self._construct_discrete_time_node(dataset, model, name)
        diff = var - data
        abs_diff = pybamm.AbsoluteValue(diff)
        expression = pybamm.DiscreteTimeSum(abs_diff**self._p) ** (1 / self._p)

        return PybammVariableMetadata(
            variable_name=name,
            expression=expression,
            parameters={},
        )

    def __name__(self):
        return f"Minkowski distance (p = {self.p})"


class SumOfPower(PybammFittingCost):
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

    def __init__(self, signal: str, p: float = 2.0):
        super().__init__(signal=signal)
        if p < 0:
            raise ValueError("The order of 'p' must be greater than 0.")
        elif not np.isfinite(p):
            raise ValueError("p = np.inf is not yet supported.")
        self._p = float(p)

    def symbolic_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammVariableMetadata:
        """Construct the variable metadata."""
        name = SumOfPower.make_unique_cost_name()

        data, var = self._construct_discrete_time_node(dataset, model, name)
        diff = var - data
        abs_diff = pybamm.AbsoluteValue(diff)
        sum_expr = abs_diff**self._p

        return PybammVariableMetadata(
            variable_name=name,
            expression=pybamm.DiscreteTimeSum(sum_expr),
            parameters={},
        )

    def __name__(self):
        return f"Sum of Power (p = {self._p})"


class BaseLikelihood(PybammFittingCost):
    """
    A base class for likelihood functions.
    These functions should be implemented as negative likelihoods for
    use within the pybop optimisation and sampling framework.
    """

    def _get_sigma_parameter(self, cost_name, parameters) -> pybamm.Parameter:
        """
        Returns the sigma node, either fixed or as an estimated parameter.
        Updates metadata if sigma is estimated.
        """

        if isinstance(self._sigma, PybopParameter) or self._sigma is None:
            sigma_name = self._sigma.name if self._sigma else f"{cost_name} (sigma)"
            sigma = pybamm.Parameter(sigma_name)
            parameters[sigma_name] = PybammParameterMetadata(
                parameter=sigma,
                default_value=self._sigma.current_value
                if self._sigma
                else 1e-2,  # Initial guess w/ ~ low noise
            )
        elif isinstance(self._sigma, float | int):
            sigma = pybamm.Scalar(self._sigma)
        else:
            sigma = self._sigma  # Assume pybamm.Parameter

        return sigma


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
        signal: str,
        sigma: float | PybopParameter | None = None,
    ):
        super().__init__(signal=signal)
        self._sigma = sigma
        self._log2pi = np.log(2 * np.pi)

    def symbolic_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammVariableMetadata:
        """Construct the variable metadata."""
        name = NegativeGaussianLogLikelihood.make_unique_cost_name()

        data, var = self._construct_discrete_time_node(dataset, model, name)
        sum_expr = pybamm.DiscreteTimeSum((data - var) ** 2.0)

        parameters = {}
        sigma = self._get_sigma_parameter(name, parameters)

        nll = (
            data.size * self._log2pi / 2.0
            + data.size * pybamm.log(sigma)
            + sum_expr / (2.0 * sigma**2.0)
        )

        return PybammVariableMetadata(
            variable_name=name,
            expression=nll,
            parameters=parameters,
        )


class ScaledCost(PybammFittingCost):
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

    def symbolic_expression(
        self,
        model: pybamm.BaseModel,
        dataset: Dataset | None = None,
    ) -> PybammVariableMetadata:
        """Construct the variable metadata."""
        name = ScaledCost.make_unique_cost_name()

        cost_metadata = self._cost.symbolic_expression(model, dataset)

        return PybammVariableMetadata(
            variable_name=name,
            expression=cost_metadata.expression
            * pybamm.Scalar(1.0)
            / len(dataset[self._cost.signal]),
            parameters=cost_metadata.parameters,
        )

    def __name__(self):
        return f"Scaled {self._cost.__name__()}"
