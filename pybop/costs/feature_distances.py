import warnings

import numpy as np
from scipy.optimize import minimize

from pybop.costs.base_cost import BaseCost
from pybop.costs.evaluation import Evaluation


def indices_of(values, target):
    roots = (np.sign(values[1:] - target) - np.sign(values[:-1] - target)).nonzero()[0]
    nearest_roots = np.unique(
        np.where(
            np.abs(values[roots] - target) < np.abs(values[roots + 1]), roots, roots + 1
        )
    )
    return nearest_roots


class FeatureDistance(BaseCost):
    """Base for defining cost functions based on comparing fit functions."""

    _supported_features = []

    def __init__(
        self,
        domain_data: np.ndarray,
        target_data: np.ndarray,
        feature: str,
        time_start: float = None,
        time_end: float = None,
    ):
        super().__init__()
        if feature not in self._supported_features:
            raise ValueError(
                "Feature '"
                + feature
                + "' not supported. Options: "
                + str(self._supported_features)
            )
        self._domain_data = domain_data
        self._target_data = target_data
        self.feature = feature
        self.time_start = time_start
        self.time_end = time_end
        self.start_index = (
            indices_of(self.domain_data, self.time_start)[0] if self.time_start else 0
        )
        self.end_index = (
            indices_of(self.domain_data, self.time_end)[0] if self.time_end else None
        )

        with warnings.catch_warnings():
            # Suppress SciPy's UserWarning about delta_grad == 0.
            warnings.simplefilter("ignore")
            self.data_fit = self._fit(
                self.domain_data[self.start_index : self.end_index],
                self.target_data[self.start_index : self.end_index],
            )

    def _inverse_fit_function(self, y, *args):
        return NotImplementedError

    def _fit_guess(self, t, y):
        return NotImplementedError

    def _feature_selection(self, fit):
        return NotImplementedError

    def _fit(self, t, y):
        """
        Uses SciPy to fit data. For numerical reasons, the fitting
        involves applying the fit function to data and comparing to identity.
        """
        t = t - t[0]
        fit_guess = self._fit_guess(t, y)
        return self._feature_selection(
            minimize(
                lambda x: np.sum((t - self._inverse_fit_function(y, *x)) ** 2) ** 0.5,
                x0=fit_guess,
                method="trust-constr",
            ).x
        )

    def __call__(
        self,
        y: np.ndarray,
        dy: np.ndarray | None = None,
    ) -> float | tuple[float, np.ndarray]:
        with warnings.catch_warnings():
            # Suppress SciPy's UserWarning about delta_grad == 0.
            warnings.simplefilter("ignore")
            error = np.abs(
                np.asarray(
                    [
                        (
                            self._fit(
                                self.domain_data[self.start_index : self.end_index],
                                y[self.start_index : self.end_index],
                            )
                            - self.data_fit
                        )
                        / self.data_fit
                    ]
                )
            )
        return Evaluation(error.item())


class SquareRootFeatureDistance(FeatureDistance):
    """
    Square-root fit cost function.

    Fits a square-root fit function and compares either its offset or
    its slope between model predictions and target data.
    """

    _supported_features = ["offset", "slope", "inverse_slope"]

    def __init__(
        self,
        domain_data: np.ndarray,
        target_data: np.ndarray,
        feature: str = "inverse_slope",
        time_start: float = None,
        time_end: float = None,
    ):
        """
        Parameters
        ----------
        domain_data : np.ndarray
            The content of the "Time [s]" entry in the used `DataSet`.
        feature : str, optional
            Set the fit parameter from the square-root fit to use for
            fitting. Possible values:
             - "offset": The value of the square-root fit at the start.
             - "slope": The prefactor of the square-root over time.
             - "inverse_slope": 1 over "slope"; may perform better.
        time_start : float, optional
            Set the time (in seconds) from which onwards the data shall be
            fitted, counted from the start of the data. Default is the start.
        time_end : float, optional
            Set the time (in seconds) until which the data shall be fitted,
            counted from the start of the data. Default is the end.
        """
        super().__init__(domain_data, target_data, feature, time_start, time_end)
        self.time_start = time_start
        self.time_end = time_end

    def _inverse_fit_function(self, y, b, c):
        """Square function to transform data for a linear fit."""
        return ((y - b) / c) ** 2

    def _fit_guess(self, t, y):
        return [y[0], (y[-1] - y[0]) / (t[-1] - t[0]) ** 0.5]

    def _feature_selection(self, fit):
        if self.feature == "offset":
            return fit[0]
        elif self.feature == "slope":
            return fit[1]
        elif self.feature == "inverse_slope":
            return 1 / fit[1]


class ExponentialFeatureDistance(FeatureDistance):
    """
    Exponential fit cost function.

    Fits an exponential and compares either its asymptote, its magnitude,
    or its timescale between model predictions and target data.
    """

    _supported_features = ["asymptote", "magnitude", "timescale", "inverse_timescale"]

    def __init__(
        self,
        domain_data: np.ndarray,
        target_data: np.ndarray,
        feature: str = "inverse_timescale",
        time_start: float = None,
        time_end: float = None,
    ):
        """
        Parameters
        ----------
        domain_data : np.ndarray
            The content of the "Time [s]" entry in the used `DataSet`.
        feature : str, optional
            Set the fit parameter from the square-root fit to use for
            fitting. Possible values:
             - "asymptote": The exponential fit value at infinite time.
             - "magnitude": The prefactor of the exponential term.
             - "timescale": The denominator in the exponential argument.
             - "inverse_timescale": 1 over "timescale"; may perform better.
        time_start : float, optional
            Set the time (in seconds) from which onwards the data shall be
            fitted, counted from the start of the data. Default is the start.
        time_end : float, optional
            Set the time (in seconds) until which the data shall be fitted,
            counted from the start of the data. Default is the end.
        """
        super().__init__(domain_data, target_data, feature, time_start, time_end)
        self.time_start = time_start
        self.time_end = time_end

    def _inverse_fit_function(self, y, b, c, d):
        """Logarithm function to transform data for a linear fit."""
        log_arg = (y - b) / c
        log_arg[log_arg <= 0] = 0.1**d
        return -d * np.log(log_arg)

    def _fit_guess(self, t, y):
        constant = y[-1]
        difference = (y[-1] - y[0]) / (t[-1] - t[0])
        second_difference = (y[-1] - 2 * y[len(y) // 2] + y[0]) / (
            t[len(t) // 2] - t[0]
        ) ** 2
        return [
            constant,
            difference**2 / second_difference,
            -difference / second_difference,
        ]

    def _feature_selection(self, fit):
        if self.feature == "asymptote":
            return fit[0]
        elif self.feature == "magnitude":
            return fit[1]
        elif self.feature == "timescale":
            return fit[2]
        elif self.feature == "inverse_timescale":
            return 1 / fit[2]
