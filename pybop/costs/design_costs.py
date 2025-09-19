import numpy as np

from pybop.costs.base_cost import BaseCost


class DesignCost(BaseCost):
    """
    Overwrites and extends `BaseCost` class for design-related cost functions.

    Inherits all parameters and attributes from ``BaseCost``.

    Note that design costs are maximised by default. Change to minimising by setting
    `minimising=True`.

    Additional Parameters
    ---------------------
    target : str
        The name of the target variable.
    """

    def __init__(self, problem, target: str):
        super().__init__(problem)
        self.minimising = False
        self.target_variable = target

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
        return y[self.target_variable][0]
