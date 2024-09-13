import numpy as np

from pybop.costs.base_cost import BaseCost


class DesignCost(BaseCost):
    """
    Overwrites and extends `BaseCost` class for design-related cost functions.

    Inherits all parameters and attributes from ``BaseCost``.

    Additional Attributes
    ---------------------
    problem : object
        The associated problem containing model and evaluation methods.
    """

    def __init__(self, problem):
        """
        Initialises the gravimetric energy density calculator with a problem.

        Parameters
        ----------
        problem : object
            The problem instance containing the model and data.
        """
        super().__init__(problem)
        self.problem = problem


class GravimetricEnergyDensity(DesignCost):
    """
    Represents the gravimetric energy density of a battery cell, calculated based
    on a normalised discharge from upper to lower voltage limits. The goal is to
    maximise the energy density, which is achieved by setting minimising = False
    in the optimiser settings.

    Inherits all parameters and attributes from ``DesignCost``.
    """

    def __init__(self, problem):
        super().__init__(problem)

    def compute(
        self,
        y: dict,
        dy: np.ndarray = None,
        calculate_grad: bool = False,
    ) -> float:
        """
        Computes the cost function for the given predictions.

        Parameters
        ----------
        y : dict
            The dictionary of predictions with keys designating the signals for fitting.
        dy : np.ndarray, optional
            The corresponding gradient with respect to the parameters for each signal.
            Note: not used in design optimisation classes.
        calculate_grad : bool, optional
            A bool condition designating whether to calculate the gradient.

        Returns
        -------
        float
            The gravimetric energy density or -infinity in case of infeasible parameters.
        """
        if not any(np.isfinite(y[signal][0]) for signal in self.signal):
            return -np.inf

        voltage, current = y["Voltage [V]"], y["Current [A]"]
        dt = y["Time [s]"][1] - y["Time [s]"][0]
        energy_density = np.trapz(voltage * current, dx=dt) / (
            3600 * self.problem.model.cell_mass()
        )

        return energy_density


class VolumetricEnergyDensity(DesignCost):
    """
    Represents the volumetric energy density of a battery cell, calculated based
    on a normalised discharge from upper to lower voltage limits. The goal is to
    maximise the energy density, which is achieved by setting minimising = False
    in the optimiser settings.

    Inherits all parameters and attributes from ``DesignCost``.
    """

    def __init__(self, problem):
        super().__init__(problem)

    def compute(
        self,
        y: dict,
        dy: np.ndarray = None,
        calculate_grad: bool = False,
    ) -> float:
        """
        Computes the cost function for the given predictions.

        Parameters
        ----------
        y : dict
            The dictionary of predictions with keys designating the signals for fitting.
        dy : np.ndarray, optional
            The corresponding gradient with respect to the parameters for each signal.
            Note: not used in design optimisation classes.
        calculate_grad : bool, optional
            A bool condition designating whether to calculate the gradient.

        Returns
        -------
        float
            The volumetric energy density or -infinity in case of infeasible parameters.
        """
        if not any(np.isfinite(y[signal][0]) for signal in self.signal):
            return -np.inf

        voltage, current = y["Voltage [V]"], y["Current [A]"]
        dt = y["Time [s]"][1] - y["Time [s]"][0]
        energy_density = np.trapz(voltage * current, dx=dt) / (
            3600 * self.problem.model.cell_volume()
        )

        return energy_density
