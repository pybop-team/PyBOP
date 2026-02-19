import warnings

import numpy as np


class BaseApplication:
    """
    A base class for PyBOP's application methods.
    """

    def check_monotonicity(self, voltage: np.ndarray) -> None:
        """
        Check if voltage data is monotonic and warn if not.

        Parameters
        ----------
        voltage : np.ndarray
            Voltage array to check for monotonicity.
        """
        is_increasing = np.all(np.diff(voltage) > 0)
        is_decreasing = np.all(np.diff(voltage) < 0)

        if not (is_increasing or is_decreasing):
            warnings.warn("OCV is not strictly monotonic.", stacklevel=2)
