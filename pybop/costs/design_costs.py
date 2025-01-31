from typing import Optional, Union

import numpy as np

from pybop.costs.base_cost import BaseCost


class DesignCost(BaseCost):
    """
    Overwrites and extends `BaseCost` class for design-related cost functions.

    Inherits all parameters and attributes from ``BaseCost``.
    """

    def __init__(self, problem):
        super().__init__(problem)
        self.minimising = False


class GravimetricEnergyDensity(DesignCost):
    """
    Calculates the gravimetric energy density (specific energy) of a battery cell,
    when applied to a normalised discharge from upper to lower voltage limits. The
    goal of maximising the energy density is achieved with self.minimising=False.

    The gravimetric energy density [Wh.kg-1] is calculated as

    .. math::
        \\frac{1}{3600 m} \\int_{t=0}^{t=T} I(t) V(t) \\mathrm{d}t

    where m is the cell mass, t is the time, T is the total time, I is the current
    and V is the voltage. The factor of 1/3600 is included to convert from seconds
    to hours.

    Inherits all parameters and attributes from ``DesignCost``.
    """

    def compute(
        self,
        y: dict,
        dy: Optional[np.ndarray] = None,
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
    Calculates the (volumetric) energy density of a battery cell, when applied to a
    normalised discharge from upper to lower voltage limits. The goal of maximising
    the energy density is achieved with self.minimising = False.

    The volumetric energy density [Wh.m-3] is calculated as

    .. math::
        \\frac{1}{3600 v} \\int_{t=0}^{t=T} I(t) V(t) \\mathrm{d}t

    where v is the cell volume, t is the time, T is the total time, I is the current
    and V is the voltage. The factor of 1/3600 is included to convert from seconds
    to hours.

    Inherits all parameters and attributes from ``DesignCost``.
    """

    def __init__(self, problem):
        super().__init__(problem)

    def compute(
        self,
        y: dict,
        dy: Optional[np.ndarray] = None,
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


class GravimetricPowerDensity(DesignCost):
    """
    Calculates the gravimetric power density (specific power) of a battery cell,
    when applied to a discharge from upper to lower voltage limits. The goal of
    maximising the power density is achieved with self.minimising=False.

    The time-averaged gravimetric power density [W.kg-1] is calculated as

    .. math::
        \\frac{1}{3600 m T} \\int_{t=0}^{t=T} I(t) V(t) \\mathrm{d}t

    where m is the cell mass, t is the time, T is the total time, I is the current
    and V is the voltage. The factor of 1/3600 is included to convert from seconds
    to hours.

    Inherits all parameters and attributes from ``DesignCost``.

    Additional parameters
    ---------------------
    target_time : int
        The length of time (seconds) over which the power should be sustained.
    """

    def __init__(self, problem, target_time: Union[int, float] = 3600):
        super().__init__(problem)
        self.target_time = target_time

    def compute(
        self,
        y: dict,
        dy: Optional[np.ndarray] = None,
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

        Returns
        -------
        float
            The gravimetric power density or -infinity in case of infeasible parameters.
        """
        if not any(np.isfinite(y[signal][0]) for signal in self.signal):
            return -np.inf

        voltage, current = y["Voltage [V]"], y["Current [A]"]
        dt = y["Time [s]"][1] - y["Time [s]"][0]
        time_averaged_power_density = np.trapz(voltage * current, dx=dt) / (
            self.target_time * 3600 * self.problem.model.cell_mass()
        )

        return time_averaged_power_density


class VolumetricPowerDensity(DesignCost):
    """
    Calculates the (volumetric) power density of a battery cell, when applied to a
    discharge from upper to lower voltage limits. The goal of maximising the power
    density is achieved with self.minimising=False.

    The time-averaged volumetric power density [W.m-3] is calculated as

    .. math::
        \\frac{1}{3600 v T} \\int_{t=0}^{t=T} I(t) V(t) \\mathrm{d}t

    where v is the cell volume, t is the time, T is the total time, I is the current
    and V is the voltage. The factor of 1/3600 is included to convert from seconds
    to hours.

    Inherits all parameters and attributes from ``DesignCost``.

    Additional parameters
    ---------------------
    target_time : int
        The length of time (seconds) over which the power should be sustained.
    """

    def __init__(self, problem, target_time: Union[int, float] = 3600):
        super().__init__(problem)
        self.target_time = target_time

    def compute(
        self,
        y: dict,
        dy: Optional[np.ndarray] = None,
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

        Returns
        -------
        float
            The volumetric power density or -infinity in case of infeasible parameters.
        """
        if not any(np.isfinite(y[signal][0]) for signal in self.signal):
            return -np.inf

        voltage, current = y["Voltage [V]"], y["Current [A]"]
        dt = y["Time [s]"][1] - y["Time [s]"][0]
        time_averaged_power_density = np.trapz(voltage * current, dx=dt) / (
            self.target_time * 3600 * self.problem.model.cell_volume()
        )

        return time_averaged_power_density
