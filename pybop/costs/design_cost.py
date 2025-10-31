import numpy as np

from pybop.costs.base_cost import BaseCost


class DesignCost(BaseCost):
    """
    Base design cost.

    Note that design costs are maximised by default. Change to minimising by setting
    the attribute `minimising=True`.

    Parameters
    ----------
    target : str
        The name of the target variable.
    """

    def __init__(self, target: str):
        super().__init__()
        self.minimising = False
        target = [target] if isinstance(target, str) else target
        self.target = target or ["Voltage [V]"]
        self.domain = "Time [s]"

    def compute(
        self,
        y: dict,
        dy: np.ndarray | None = None,
    ) -> float:
        """
        Returns the value of the cost variable.

        Parameters
        ----------
        y : dict
            The dictionary of predictions with keys designating the output variables for fitting.
        dy : np.ndarray, optional
            The corresponding gradient with respect to the parameters for each output variable.
            Note: not used in design optimisation classes.

        Returns
        -------
        float
            The value of the output variable.
        """
        if not self.verify_prediction(y):
            return self.failure(dy)

        return y[self.target[0]][-1]

    def verify_prediction(self, y: dict):
        """
        Verify that the prediction matches the target data.

        Parameters
        ----------
        y : dict
            A dictionary of predictions with keys designating the output variables for fitting.

        Returns
        -------
        bool
            True if the prediction matches the target data, otherwise False.
        """
        if not all(np.isfinite(y[var]) for var in self.target):
            return False

        return True
