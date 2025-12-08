import numpy as np
import pybamm

from pybop._dataset import Dataset
from pybop._utils import FailedSolution
from pybop.costs.base_cost import BaseCost
from pybop.parameters.parameter import Inputs
from pybop.simulators.solution import Solution


class ErrorMeasure(BaseCost):
    """
    Base fitting cost (error measure).

    This class is intended to be subclassed to create specific cost functions
    for evaluating model predictions against a set of data. The cost function
    quantifies the goodness-of-fit between model predictions and the target
    data, with a lower cost value indicating a better fit.

    Parameters
    ----------
    dataset : pybop.Dataset
        Dataset object containing the target data.
    target : list[str]
        The name(s) of the target variable(s).
    weighting : Union[str, np.ndarray], optional
        The type of weighting to use when taking the sum or mean of the error
        measure.

    Attributes
    ----------
    dataset : dictionary
        The dictionary from a Dataset object containing the target data.
    domain : str
        The name of the domain (default: "Time [s]").
    domain_data : np.ndarray
        The domain points in the dataset.
    n_domain_data : int
        The number of domain points.
    target_data : np.ndarray
        The target values of the output variables.
    """

    def __init__(
        self,
        dataset: Dataset,
        target: str | list[str] = None,
        weighting: float | str | np.ndarray = None,
    ):
        super().__init__()
        self.target = [target] if isinstance(target, str) else target or ["Voltage [V]"]
        self.set_target(dataset)
        self.set_weighting(weighting)

    def set_target(self, dataset: Dataset):
        """Set the target data from a pybop.Dataset."""
        if not isinstance(dataset, Dataset):
            raise ValueError("Dataset must be a pybop.Dataset object.")
        self.domain = dataset.domain

        # Check that the dataset contains necessary variables
        dataset.check(domain=self.domain, signal=self.target)
        self._dataset = dataset.data

        # Unpack domain and target data
        self._domain_data = self._dataset[self.domain]
        self.n_data = len(self._domain_data)

        self._domain_data = dataset[self.domain]
        self._target_data = {var: dataset[var] for var in self.target}
        self.n_outputs = len(self.target)

    def set_weighting(self, weighting: float | str | np.ndarray):
        if weighting == "equal" or weighting is None:
            self.weighting = 1.0
        elif weighting == "domain":
            # Normalise the residuals by the domain spacing (for a uniform domain,
            # this is the same as a uniform weighting)
            domain_data = self._domain_data
            domain_spacing = domain_data[1:] - domain_data[:-1]
            mean_spacing = np.mean(domain_spacing)
            self.weighting = np.concatenate(
                (
                    [(mean_spacing + domain_spacing[0]) / 2],
                    (domain_spacing[1:] + domain_spacing[:-1]) / 2,
                    [(domain_spacing[-1] + mean_spacing) / 2],
                )
            ) * ((len(domain_data) - 1) / (domain_data[-1] - domain_data[0]))
        else:
            self.weighting = np.asarray(weighting)

    def evaluate(
        self,
        sol: Solution | pybamm.Solution | FailedSolution,
        inputs: Inputs | None = None,
        calculate_sensitivities: bool = False,
    ) -> float | tuple[float, np.ndarray]:
        """
        Computes the cost function for the given predictions.

        Parameters
        ----------
        sol : pybop.Solution | pybamm.Solution
            The simulation result.
        inputs : Inputs, optional
            Input parameters (default: None).
        calculate_sensitivities : bool
            Whether to also return the sensitivities (default: False).

        Returns
        -------
        np.float64 or tuple[np.float64, np.ndarray[np.float64]]
            If the solution has sensitivities, returns a tuple containing the cost (float) and the
            gradient with dimension (len(parameters)), otherwise returns only the cost.
        """
        # Return failure cost if the solution failed
        if isinstance(sol, FailedSolution):
            return self.failure(calculate_sensitivities)

        if not isinstance(sol, (Solution, pybamm.Solution)):
            raise ValueError(
                f"sol must be a pybop.Solution object, got type {type(sol)} with value {sol}."
            )

        # Early return if the prediction is not verified
        if not self.verify_prediction(sol):
            return self.failure(calculate_sensitivities)

        # Compute the residual for all output variables
        r = np.asarray([sol[var].data - self._target_data[var] for var in self.target])

        # Extract the sensitivities for all output variables and parameters
        dy = self.stack_sensitivities(sol) if calculate_sensitivities else None

        return self.__call__(r=r, dy=dy, inputs=inputs)

    def verify_prediction(self, sol: Solution):
        """
        Verify that the prediction matches the target data.

        Parameters
        ----------
        sol : pybop.Solution | pybamm.Solution
            The simulation result.

        Returns
        -------
        bool
            True if the prediction matches the target data, otherwise False.
        """
        if any(
            len(sol[key].data) != len(self._target_data.get(key, []))
            for key in self.target
        ):
            return False

        return True

    def __call__(
        self,
        r: np.ndarray,
        dy: np.ndarray | None = None,
        inputs: Inputs | None = None,
    ) -> float | tuple[float, np.ndarray]:
        """
        Computes the cost function for the given predictions.

        Parameters
        ----------
        r : np.ndarray
            The residual difference between the model prediction and the target. The
            dimensions of r are (len(target), len(domain_data)).
        dy : np.ndarray, optional
            The corresponding gradient with respect to the parameters for each output variable.
            The dimensions of dy are (len(parameters), len(target), len(domain_data)).

        Returns
        -------
        np.float64 or tuple[np.float64, np.ndarray[np.float64]]
            If dy is not None, returns a tuple containing the cost (float) and the
            gradient with dimension (len(parameters)), otherwise returns only the cost.
        """
        raise NotImplementedError

    @property
    def target_data(self):
        return self._target_data

    @property
    def domain_data(self):
        return self._domain_data

    @domain_data.setter
    def domain_data(self, domain_data):
        self._domain_data = domain_data

    @property
    def dataset(self):
        return self._dataset


class MeanSquaredError(ErrorMeasure):
    """
    Mean square error (MSE) cost function.

    Computes the mean square error between model predictions and the target
    data, providing a measure of the differences between predicted values and
    observed values.
    """

    def __call__(
        self,
        r: np.ndarray,
        dy: np.ndarray | None = None,
        inputs: Inputs | None = None,
    ) -> float | tuple[float, np.ndarray]:
        e = np.mean((np.abs(r) ** 2) * self.weighting)

        if dy is not None:
            de = 2 * np.mean((r * self.weighting) * dy, axis=(1, 2))
            return e, de

        return e


class RootMeanSquaredError(ErrorMeasure):
    """
    Root mean square error (RMSE) cost function.

    Computes the root mean square error between model predictions and the target
    data, providing a measure of the differences between predicted values and
    observed values.
    """

    def __call__(
        self,
        r: np.ndarray,
        dy: np.ndarray | None = None,
        inputs: Inputs | None = None,
    ) -> float | tuple[float, np.ndarray]:
        e = np.sqrt(np.mean((np.abs(r) ** 2) * self.weighting))

        if dy is not None:
            de = np.mean((r * self.weighting) * dy, axis=(1, 2)) / (
                e + np.finfo(float).eps
            )
            return e, de

        return e


class MeanAbsoluteError(ErrorMeasure):
    """
    Mean absolute error (MAE) cost function.

    Computes the mean absolute error (MAE) between model predictions
    and target data. The MAE is a measure of the average magnitude
    of errors in a set of predictions, without considering their direction.
    """

    def __call__(
        self,
        r: np.ndarray,
        dy: np.ndarray | None = None,
        inputs: Inputs | None = None,
    ) -> float | tuple[float, np.ndarray]:
        e = np.mean(np.abs(r) * self.weighting)

        if dy is not None:
            de = np.mean((np.sign(r) * self.weighting) * dy, axis=(1, 2))
            return e, de

        return e


class SumSquaredError(ErrorMeasure):
    """
    Sum of squared error (SSE) cost function.

    Computes the sum of the squares of the differences between model predictions
    and target data, which serves as a measure of the total error between the
    predicted and observed values.
    """

    def __call__(
        self,
        r: np.ndarray,
        dy: np.ndarray | None = None,
        inputs: Inputs | None = None,
    ) -> float | tuple[float, np.ndarray]:
        e = np.sum(np.abs(r) ** 2 * self.weighting)

        if dy is not None:
            de = 2 * np.sum((r * self.weighting) * dy, axis=(1, 2))
            return e, de

        return e


class Minkowski(ErrorMeasure):
    """
    The Minkowski distance is a generalisation of several distance metrics,
    including the Euclidean and Manhattan distances. It is defined as:

    .. math::
        L_p(x, y) = ( \\sum_i |x_i - y_i|^p )^(1/p)

    where p > 0 is the order of the Minkowski distance. For p ≥ 1, the
    Minkowski distance is a metric. For 0 < p < 1, it is not a metric, as it
    does not satisfy the triangle inequality, although a metric can be
    obtained by removing the (1/p) exponent.

    Special cases:

    * p = 1: Manhattan distance
    * p = 2: Euclidean distance
    * p → ∞: Chebyshev distance (not implemented as yet)

    This class implements the Minkowski distance as a cost function for
    optimisation problems, allowing for flexible distance-based optimisation
    across various problem domains.

    Additional Attributes
    ---------------------
    p : float, optional
        The order of the Minkowski distance.
    """

    def __init__(
        self,
        dataset: Dataset,
        p: float = 2.0,
        target: str | list[str] = None,
        weighting: str | np.ndarray = None,
    ):
        super().__init__(dataset=dataset, target=target, weighting=weighting)
        if p < 0:
            raise ValueError(
                "The order of the Minkowski distance must be greater than 0."
            )
        elif not np.isfinite(p):
            raise ValueError(
                "For p = infinity, an implementation of the Chebyshev distance is required."
            )
        self.p = float(p)

    def __call__(
        self,
        r: np.ndarray,
        dy: np.ndarray | None = None,
        inputs: Inputs | None = None,
    ) -> float | tuple[float, np.ndarray]:
        e = np.sum((np.abs(r) ** self.p) * self.weighting) ** (1 / self.p)

        if dy is not None:
            de = np.sum(
                ((np.sign(r) * np.abs(r) ** (self.p - 1)) * self.weighting) * dy,
                axis=(1, 2),
            ) / (e ** (self.p - 1) + np.finfo(float).eps)
            return e, de

        return e

    @property
    def __name__(self):
        return f"Minkowski distance (p = {self.p})"


class SumOfPower(ErrorMeasure):
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

    Additional Attributes
    ---------------------
    p : float, optional
        The power order for Sum of Power.
    """

    def __init__(
        self,
        dataset: Dataset,
        p: float = 2.0,
        target: str | list[str] = None,
        weighting: str | np.ndarray = None,
    ):
        super().__init__(dataset=dataset, target=target, weighting=weighting)
        if p < 0:
            raise ValueError("The order of 'p' must be greater than 0.")
        elif not np.isfinite(p):
            raise ValueError("p = np.inf is not yet supported.")
        self.p = float(p)

    def __call__(
        self,
        r: np.ndarray,
        dy: np.ndarray | None = None,
        inputs: Inputs | None = None,
    ) -> float | tuple[float, np.ndarray]:
        e = np.sum((np.abs(r) ** self.p) * self.weighting)

        if dy is not None:
            de = self.p * np.sum(
                ((np.sign(r) * np.abs(r) ** (self.p - 1)) * self.weighting) * dy,
                axis=(1, 2),
            )
            return e, de

        return e

    def __name__(self):
        return f"Sum of Power (p = {self.p})"
