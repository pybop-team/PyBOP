import numpy as np
import warnings

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
        super().__init__(problem)
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
        self.parameter_set = problem._model._parameter_set
        self.update_simulation_data(problem.x0)

    def update_simulation_data(self, initial_conditions):
        """
        Updates the simulation data based on the initial conditions.

        Parameters
        ----------
        initial_conditions : array
            The initial conditions for the simulation.
        """
        if self.update_capacity:
            self.problem._model.approximate_capacity(self.problem.x0)
        solution = self.problem.evaluate(initial_conditions)
        self.problem._time_data = solution[:, -1]
        self.problem._target = solution[:, 0:-1]
        self.dt = solution[1, -1] - solution[0, -1]

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
    maximise the energy density, which is achieved by minimizing the negative energy
    density reported by this class.

    Inherits all parameters and attributes from ``DesignCost``.
    """

    def __init__(self, problem, update_capacity=False):
        super().__init__(problem, update_capacity)

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
            The negative gravimetric energy density or infinity in case of infeasible parameters.
        """
        try:
            with warnings.catch_warnings():
                # Convert UserWarning to an exception
                warnings.filterwarnings("error", category=UserWarning)

                if self.update_capacity:
                    self.problem._model.approximate_capacity(x)
                solution = self.problem.evaluate(x)

                voltage, current = solution[:, 0], solution[:, 1]
                negative_energy_density = -np.trapz(voltage * current, dx=self.dt) / (
                    3600 * self.problem._model.cell_mass(self.parameter_set)
                )

                return negative_energy_density

        except UserWarning as e:
            print(f"Ignoring this sample due to: {e}")
            return np.inf

        except Exception as e:
            print(f"An error occurred during the evaluation: {e}")
            return np.inf


class VolumetricEnergyDensity(DesignCost):
    """
    Represents the volumetric energy density of a battery cell, calculated based
    on a normalised discharge from upper to lower voltage limits. The goal is to
    maximise the energy density, which is achieved by minimizing the negative energy
    density reported by this class.

    Inherits all parameters and attributes from ``DesignCost``.
    """

    def __init__(self, problem, update_capacity=False):
        super().__init__(problem, update_capacity)

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
            The negative volumetric energy density or infinity in case of infeasible parameters.
        """
        try:
            with warnings.catch_warnings():
                # Convert UserWarning to an exception
                warnings.filterwarnings("error", category=UserWarning)

                if self.update_capacity:
                    self.problem._model.approximate_capacity(x)
                solution = self.problem.evaluate(x)

                voltage, current = solution[:, 0], solution[:, 1]
                negative_energy_density = -np.trapz(voltage * current, dx=self.dt) / (
                    3600 * self.problem._model.cell_volume(self.parameter_set)
                )

                return negative_energy_density

        except UserWarning as e:
            print(f"Ignoring this sample due to: {e}")
            return np.inf

        except Exception as e:
            print(f"An error occurred during the evaluation: {e}")
            return np.inf
