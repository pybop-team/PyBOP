from typing import Union

import numpy as np
from ep_bolfi.utility.fitting_functions import fit_sqrt
from ep_bolfi.utility.preprocessing import find_occurrences

from pybop import BaseCost


class SquareRootFitOffset(BaseCost):
    """
    Square-root fit cost function.

    Fits a square-root fit function and compares its offset between
    model predictions and target data.
    """

    def __init__(
        self,
        problem,
        time_start: float = None,
        time_end: float = None,
        threshold: float = 0.95,
    ):
        """
        Parameters
        ----------
        problem : Inherited from pybop.BaseCost, see there.
        time_start : float, optional
            Set the time (in seconds) from which onwards the data shall be
            fitted, counted from the start of the data. Default is the start.
        time_end : float, optional
            Set the time (in seconds) until which the data shall be fitted,
            counted from the start of the data. Default is the end.
        threshold : float, optional
            A threshold for the R²-value of the square-root fit. If the initial
            fit does not reach this threshold, data points at the end of the
            time interval are culled until the fit to the remaining data
            reaches it. Set to a value between 0 and 1, excluding 1 itself.
        """
        super().__init__(problem)
        self.time_start = time_start
        self.time_end = time_end
        self.threshold = threshold

    def compute(
        self,
        y: dict,
        dy: np.ndarray = None,
        calculate_grad: bool = False,
    ) -> Union[float, tuple[float, np.ndarray]]:
        # Gradient not available for fitting function parameters.
        if calculate_grad:
            raise ValueError("Square-root fit cost does not support gradients.")
        # Early return if the prediction is not verified.
        if not self.verify_prediction(y):
            return np.inf
        # Calculate square-root fit function.
        if self.time_start:
            start_index = find_occurrences(y["Time [s]"], self.time_start)[0]
        else:
            start_index = 0
        if self.time_end:
            end_index = find_occurrences(y["Time [s]"], self.time_end)[0]
        else:
            end_index = None
        error = np.asarray(
            [
                fit_sqrt(
                    y["Time [s]"][start_index:end_index],
                    y[signal][start_index:end_index],
                    self.threshold,
                )[2][0]
                - fit_sqrt(
                    self._target["Time [s]"][start_index:end_index],
                    self._target[signal][start_index:end_index],
                    self.threshold,
                )[2][0]
                for signal in self.signal
            ]
        )
        return error.item() if self.n_outputs == 1 else np.sum(error)


class SquareRootFitInverseSlope(BaseCost):
    """
    Square-root fit cost function.

    Fits a square-root fit function and compares its inverse slope
    between model predictions and target data.
    """

    def __init__(
        self,
        problem,
        time_start: float = None,
        time_end: float = None,
        threshold: float = 0.95,
    ):
        """
        Parameters
        ----------
        problem : Inherited from pybop.BaseCost, see there.
        time_start : float, optional
            Set the time (in seconds) from which onwards the data shall be
            fitted, counted from the start of the data. Default is the start.
        time_end : float, optional
            Set the time (in seconds) until which the data shall be fitted,
            counted from the start of the data. Default is the end.
        threshold : float, optional
            A threshold for the R²-value of the square-root fit. If the initial
            fit does not reach this threshold, data points at the end of the
            time interval are culled until the fit to the remaining data
            reaches it. Set to a value between 0 and 1, excluding 1 itself.
        """
        super().__init__(problem)
        self.time_start = time_start
        self.time_end = time_end
        self.threshold = threshold

    def compute(
        self,
        y: dict,
        dy: np.ndarray = None,
        calculate_grad: bool = False,
    ) -> Union[float, tuple[float, np.ndarray]]:
        # Gradient not available for fitting function parameters.
        if calculate_grad:
            raise ValueError("Square-root fit cost does not support gradients.")
        # Early return if the prediction is not verified.
        if not self.verify_prediction(y):
            return np.inf
        # Calculate square-root fit function.
        if self.time_start:
            start_index = find_occurrences(y["Time [s]"], self.time_start)[0]
        else:
            start_index = 0
        if self.time_end:
            end_index = find_occurrences(y["Time [s]"], self.time_end)[0]
        else:
            end_index = None
        error = np.asarray(
            [
                1
                / fit_sqrt(
                    y["Time [s]"][start_index:end_index],
                    y[signal][start_index:end_index],
                    self.threshold,
                )[2][1]
                - 1
                / fit_sqrt(
                    self._target["Time [s]"][start_index:end_index],
                    self._target[signal][start_index:end_index],
                    self.threshold,
                )[2][1]
                for signal in self.signal
            ]
        )
        return error.item() if self.n_outputs == 1 else np.sum(error)
