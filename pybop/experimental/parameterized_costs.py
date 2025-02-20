from typing import Union

import numpy as np
from scipy.optimize import minimize

from pybop import BaseCost


class ParameterizedCost(BaseCost):
    """Base for defining cost functions based on fit parameters."""

    def __init__(self, problem, time_data: np.ndarray, feature: str):
        super().__init__(problem)
        if feature not in self._supported_features:
            raise ValueError(
                "Feature '"
                + feature
                + "' not supported. Options: "
                + str(self._supported_features)
            )
        self.feature = feature
        self.time_data = time_data

    @property
    def _supported_features(self):
        raise NotImplementedError

    def _inverse_fit_function(self, y, *args):
        raise NotImplementedError

    def _fit_guess(self, t, y):
        raise NotImplementedError

    def _feature_selection(self, fit):
        raise NotImplementedError

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

    def compute(
        self,
        y: dict,
        dy: np.ndarray = None,
        calculate_grad: bool = False,
    ) -> Union[float, tuple[float, np.ndarray]]:
        from ep_bolfi.utility.preprocessing import find_occurrences

        time_domain_data = self.problem.domain_data  # y["Time [s]"]
        # Gradient not available for fitting function parameters.
        if calculate_grad:
            raise ValueError("Square-root fit cost does not support gradients.")
        # Early return if the prediction is not verified.
        if not self.verify_prediction(y):
            return np.inf
        # Calculate square-root fit function.
        if self.time_start:
            start_index = find_occurrences(time_domain_data, self.time_start)[0]
        else:
            start_index = 0
        if self.time_end:
            end_index = find_occurrences(time_domain_data, self.time_end)[0]
        else:
            end_index = None
        error = np.abs(
            np.asarray(
                [
                    self._fit(
                        time_domain_data[start_index:end_index],
                        y[signal][start_index:end_index],
                    )
                    - self._fit(
                        self.time_data[start_index:end_index],
                        self.target[signal][start_index:end_index],
                    )
                    for signal in self.signal
                ]
            )
        )
        return error.item() if self.n_outputs == 1 else np.sum(error)


class SquareRootFit(ParameterizedCost):
    """
    Square-root fit cost function.

    Fits a square-root fit function and compares either its offset or
    its slope between model predictions and target data.
    """

    def __init__(
        self,
        problem,
        time_data: np.ndarray,
        feature: str = "inverse_slope",
        time_start: float = None,
        time_end: float = None,
    ):
        """
        Parameters
        ----------
        problem : Inherited from pybop.BaseCost, see there.
        time_data : np.ndarray
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
        super().__init__(problem, time_data, feature)
        self.time_start = time_start
        self.time_end = time_end

    @property
    def _supported_features(self):
        return ["offset", "slope", "inverse_slope"]

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


class ExponentialFit(ParameterizedCost):
    """
    Exponential fit cost function.

    Fits an exponential and compares either its asymptote, its magnitude,
    or its timescale between model predictions and target data.
    """

    def __init__(
        self,
        problem,
        time_data: np.ndarray,
        feature: str = "inverse_timescale",
        time_start: float = None,
        time_end: float = None,
    ):
        """
        Parameters
        ----------
        problem : Inherited from pybop.BaseCost, see there.
        time_data : np.ndarray
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
        super().__init__(problem, time_data, feature)
        self.time_start = time_start
        self.time_end = time_end

    @property
    def _supported_features(self):
        return ["asymptote", "magnitude", "timescale", "inverse_timescale"]

    def _inverse_fit_function(self, y, b, c, d):
        """Logarithm function to transform data for a linear fit."""
        log_arg = (y - b) / c
        log_arg[log_arg <= 0] = 0.1**d
        return np.log(log_arg) / d

    def _fit_guess(self, t, y):
        return [y[-1], y[-1] - y[len(y) // 10], 1 / (t[-1] - t[len(t) // 10])]

    def _feature_selection(self, fit):
        if self.feature == "asymptote":
            return fit[0]
        elif self.feature == "magnitude":
            return fit[1]
        elif self.feature == "timescale":
            return fit[2]
        elif self.feature == "inverse_timescale":
            return 1 / fit[2]
