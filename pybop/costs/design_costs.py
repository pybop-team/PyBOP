import warnings

import numpy as np

from pybop import is_numeric
from pybop.costs.base_cost import BaseCost


class DesignCost(BaseCost):
    """
    Overwrites and extends `BaseCost` class for design-related cost functions.

    Inherits all parameters and attributes from ``BaseCost``.

    Additional Attributes
    ---------------------
    problem : object
        The associated problem containing model and evaluation methods.
    parameter_set : object)
        The set of parameters from the problem's model.
    dt : float
        The time step size used in the simulation.
    """

    def __init__(self, problem, update_capacity=False):
        """
        Initialises the gravimetric energy density calculator with a problem.

        Parameters
        ----------
        problem : object
            The problem instance containing the model and data.
        """
        super(DesignCost, self).__init__(problem)
        self.problem = problem
        if update_capacity is True:
            nominal_capacity_warning = (
                "The nominal capacity is approximated for each iteration."
            )
        else:
            nominal_capacity_warning = (
                "The nominal capacity is fixed at the initial model value."
            )
        warnings.warn(nominal_capacity_warning, UserWarning)
        self.update_capacity = update_capacity
        self.parameter_set = problem.model.parameter_set
        self.update_simulation_data(self.x0)

    def update_simulation_data(self, x0):
        """
        Updates the simulation data based on the initial parameter values.

        Parameters
        ----------
        x0 : array
            The initial parameter values for the simulation.
        """
        if self.update_capacity:
            self.problem.model.approximate_capacity(x0)
        solution = self.problem.evaluate(x0)

        if "Time [s]" not in solution:
            raise ValueError("The solution does not contain time data.")
        self.problem._time_data = solution["Time [s]"]
        self.problem._target = {key: solution[key] for key in self.problem.signal}
        self.dt = solution["Time [s]"][1] - solution["Time [s]"][0]

    def _evaluate(self, x, grad=None):
        """
        Computes the value of the cost function.

        This method must be implemented by subclasses.

        Parameters
        ----------
        x : array
            The parameter set for which to compute the cost.
        grad : array, optional
            Gradient information, not used in this method.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the subclass.
        """
        raise NotImplementedError


class GravimetricEnergyDensity(DesignCost):
    """
    Represents the gravimetric energy density of a battery cell, calculated based
    on a normalised discharge from upper to lower voltage limits. The goal is to
    maximise the energy density, which is achieved by setting minimising = False
    in the optimiser settings.

    Inherits all parameters and attributes from ``DesignCost``.
    """

    def __init__(self, problem, update_capacity=False):
        super(GravimetricEnergyDensity, self).__init__(problem, update_capacity)

    def _evaluate(self, x, grad=None):
        """
        Computes the cost function for the energy density.

        Parameters
        ----------
        x : array
            The parameter set for which to compute the cost.
        grad : array, optional
            Gradient information, not used in this method.

        Returns
        -------
        float
            The gravimetric energy density or -infinity in case of infeasible parameters.
        """
        if not all(is_numeric(i) for i in x):
            raise ValueError("Input must be a numeric array.")

        try:
            with warnings.catch_warnings():
                # Convert UserWarning to an exception
                warnings.filterwarnings("error", category=UserWarning)

                if self.update_capacity:
                    self.problem.model.approximate_capacity(x)
                solution = self.problem.evaluate(x)

                voltage, current = solution["Voltage [V]"], solution["Current [A]"]
                energy_density = np.trapz(voltage * current, dx=self.dt) / (
                    3600 * self.problem.model.cell_mass(self.parameter_set)
                )

                return energy_density

        # Catch infeasible solutions and return infinity
        except UserWarning as e:
            print(f"Ignoring this sample due to: {e}")
            return -np.inf

        # Catch any other exception and return infinity
        except Exception as e:
            print(f"An error occurred during the evaluation: {e}")
            return -np.inf


class VolumetricEnergyDensity(DesignCost):
    """
    Represents the volumetric energy density of a battery cell, calculated based
    on a normalised discharge from upper to lower voltage limits. The goal is to
    maximise the energy density, which is achieved by setting minimising = False
    in the optimiser settings.

    Inherits all parameters and attributes from ``DesignCost``.
    """

    def __init__(self, problem, update_capacity=False):
        super(VolumetricEnergyDensity, self).__init__(problem, update_capacity)

    def _evaluate(self, x, grad=None):
        """
        Computes the cost function for the energy density.

        Parameters
        ----------
        x : array
            The parameter set for which to compute the cost.
        grad : array, optional
            Gradient information, not used in this method.

        Returns
        -------
        float
            The volumetric energy density or -infinity in case of infeasible parameters.
        """
        if not all(is_numeric(i) for i in x):
            raise ValueError("Input must be a numeric array.")
        try:
            with warnings.catch_warnings():
                # Convert UserWarning to an exception
                warnings.filterwarnings("error", category=UserWarning)

                if self.update_capacity:
                    self.problem.model.approximate_capacity(x)
                solution = self.problem.evaluate(x)

                voltage, current = solution["Voltage [V]"], solution["Current [A]"]
                energy_density = np.trapz(voltage * current, dx=self.dt) / (
                    3600 * self.problem.model.cell_volume(self.parameter_set)
                )

                return energy_density

        # Catch infeasible solutions and return infinity
        except UserWarning as e:
            print(f"Ignoring this sample due to: {e}")
            return -np.inf

        # Catch any other exception and return infinity
        except Exception as e:
            print(f"An error occurred during the evaluation: {e}")
            return -np.inf
