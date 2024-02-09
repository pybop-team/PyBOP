import numpy as np
import warnings

from pybop.observers.observer import Observer


class BaseCost:
    """
    Base class for defining cost functions.

    This class is intended to be subclassed to create specific cost functions
    for evaluating model predictions against a set of data. The cost function
    quantifies the goodness-of-fit between the model predictions and the
    observed data, with a lower cost value indicating a better fit.

    Parameters
    ----------
    problem : object
        A problem instance containing the data and functions necessary for
        evaluating the cost function.
    _target : array-like
        An array containing the target data to fit.
    x0 : array-like
        The initial guess for the model parameters.
    bounds : tuple
        The bounds for the model parameters.
    n_parameters : int
        The number of parameters in the model.
    n_outputs : int
        The number of outputs in the model.
    """

    def __init__(self, problem):
        self.problem = problem
        if problem is not None:
            self._target = problem._target
            self.x0 = problem.x0
            self.bounds = problem.bounds
            self.n_parameters = problem.n_parameters
            self.n_outputs = problem.n_outputs

    def __call__(self, x, grad=None):
        """
        Call the evaluate function for a given set of parameters.

        Parameters
        ----------
        x : array-like
            The parameters for which to evaluate the cost.
        grad : array-like, optional
            An array to store the gradient of the cost function with respect
            to the parameters.

        Returns
        -------
        float
            The calculated cost function value.

        Raises
        ------
        ValueError
            If an error occurs during the calculation of the cost.
        """
        try:
            return self._evaluate(x, grad)

        except Exception as e:
            raise ValueError(f"Error in cost calculation: {e}")

    def _evaluate(self, x, grad=None):
        """
        Calculate the cost function value for a given set of parameters.

        This method must be implemented by subclasses.

        Parameters
        ----------
        x : array-like
            The parameters for which to evaluate the cost.
        grad : array-like, optional
            An array to store the gradient of the cost function with respect
            to the parameters.

        Returns
        -------
        float
            The calculated cost function value.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the subclass.
        """
        raise NotImplementedError

    def evaluateS1(self, x):
        """
        Call _evaluateS1 for a given set of parameters.

        Parameters
        ----------
        x : array-like
            The parameters for which to compute the cost and gradient.

        Returns
        -------
        tuple
            A tuple containing the cost and the gradient. The cost is a float,
            and the gradient is an array-like of the same length as `x`.

        Raises
        ------
        ValueError
            If an error occurs during the calculation of the cost or gradient.
        """
        try:
            return self._evaluateS1(x)

        except Exception as e:
            raise ValueError(f"Error in cost calculation: {e}")

    def _evaluateS1(self, x):
        """
        Compute the cost and its gradient with respect to the parameters.

        Parameters
        ----------
        x : array-like
            The parameters for which to compute the cost and gradient.

        Returns
        -------
        tuple
            A tuple containing the cost and the gradient. The cost is a float,
            and the gradient is an array-like of the same length as `x`.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the subclass.
        """
        raise NotImplementedError


class RootMeanSquaredError(BaseCost):
    """
    Root mean square error cost function.

    Computes the root mean square error between model predictions and the target
    data, providing a measure of the differences between predicted values and
    observed values.

    Inherits all parameters and attributes from ``BaseCost``.

    """

    def __init__(self, problem):
        super(RootMeanSquaredError, self).__init__(problem)

    def _evaluate(self, x, grad=None):
        """
        Calculate the root mean square error for a given set of parameters.

        Parameters
        ----------
        x : array-like
            The parameters for which to evaluate the cost.
        grad : array-like, optional
            An array to store the gradient of the cost function with respect
            to the parameters.

        Returns
        -------
        float
            The root mean square error.

        """

        prediction = self.problem.evaluate(x)

        if len(prediction) < len(self._target):
            return np.float64(np.inf)  # simulation stopped early
        else:
            return np.sqrt(np.mean((prediction - self._target) ** 2))


class SumSquaredError(BaseCost):
    """
    Sum of squared errors cost function.

    Computes the sum of the squares of the differences between model predictions
    and target data, which serves as a measure of the total error between the
    predicted and observed values.

    Inherits all parameters and attributes from ``BaseCost``.

    Additional Attributes
    ---------------------
    _de : float
        The gradient of the cost function to use if an error occurs during
        evaluation. Defaults to 1.0.

    """

    def __init__(self, problem):
        super(SumSquaredError, self).__init__(problem)

        # Default fail gradient
        self._de = 1.0

    def _evaluate(self, x, grad=None):
        """
        Calculate the sum of squared errors for a given set of parameters.

        Parameters
        ----------
        x : array-like
            The parameters for which to evaluate the cost.
        grad : array-like, optional
            An array to store the gradient of the cost function with respect
            to the parameters.

        Returns
        -------
        float
            The sum of squared errors.
        """
        prediction = self.problem.evaluate(x)

        if len(prediction) < len(self._target):
            return np.float64(np.inf)  # simulation stopped early
        else:
            return np.sum(
                (np.sum(((prediction - self._target) ** 2), axis=0)),
                axis=0,
            )

    def _evaluateS1(self, x):
        """
        Compute the cost and its gradient with respect to the parameters.

        Parameters
        ----------
        x : array-like
            The parameters for which to compute the cost and gradient.

        Returns
        -------
        tuple
            A tuple containing the cost and the gradient. The cost is a float,
            and the gradient is an array-like of the same length as `x`.

        Raises
        ------
        ValueError
            If an error occurs during the calculation of the cost or gradient.
        """
        y, dy = self.problem.evaluateS1(x)
        if len(y) < len(self._target):
            e = np.float64(np.inf)
            de = self._de * np.ones(self.n_parameters)
        else:
            dy = dy.reshape(
                (
                    self.problem.n_time_data,
                    self.n_outputs,
                    self.n_parameters,
                )
            )
            r = y - self._target
            e = np.sum(np.sum(r**2, axis=0), axis=0)
            de = 2 * np.sum(np.sum((r.T * dy.T), axis=2), axis=1)

        return e, de

    def set_fail_gradient(self, de):
        """
        Set the fail gradient to a specified value.

        The fail gradient is used if an error occurs during the calculation
        of the gradient. This method allows updating the default gradient value.

        Parameters
        ----------
        de : float
            The new fail gradient value to be used.
        """
        de = float(de)
        self._de = de


class GravimetricEnergyDensity(BaseCost):
    """
    Represents the gravimetric energy density of a battery cell, calculated based
    on a normalised discharge from upper to lower voltage limits. The goal is to
    maximise the energy density, which is achieved by minimizing the negative energy
    density reported by this class.

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


class VolumetricEnergyDensity(BaseCost):
    """
    Represents the volumetric energy density of a battery cell, calculated based
    on a normalised discharge from upper to lower voltage limits. The goal is to
    maximise the energy density, which is achieved by minimizing the negative energy
    density reported by this class.

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
        Initialises the volumetric energy density calculator with a problem.

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


class ObserverCost(BaseCost):
    """
    Observer cost function.

    Computes the cost function for an observer model, which is log likelihood
    of the data points given the model parameters.

    Inherits all parameters and attributes from ``BaseCost``.

    """

    def __init__(self, observer: Observer):
        super().__init__(problem=observer)
        self._observer = observer

    def _evaluate(self, x, grad=None):
        """
        Calculate the observer cost for a given set of parameters.

        Parameters
        ----------
        x : array-like
            The parameters for which to evaluate the cost.
        grad : array-like, optional
            An array to store the gradient of the cost function with respect
            to the parameters.

        Returns
        -------
        float
            The observer cost (negative of the log likelihood).
        """
        inputs = {key: x[i] for i, key in enumerate(self._observer._model.fit_keys)}
        log_likelihood = self._observer.log_likelihood(
            self._target, self._observer.time_data(), inputs
        )
        return -log_likelihood

    def evaluateS1(self, x):
        """
        Compute the cost and its gradient with respect to the parameters.

        Parameters
        ----------
        x : array-like
            The parameters for which to compute the cost and gradient.

        Returns
        -------
        tuple
            A tuple containing the cost and the gradient. The cost is a float,
            and the gradient is an array-like of the same length as `x`.

        Raises
        ------
        ValueError
            If an error occurs during the calculation of the cost or gradient.
        """
        raise NotImplementedError
