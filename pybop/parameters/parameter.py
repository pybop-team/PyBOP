from collections import OrderedDict
from typing import Dict, List

import numpy as np


class Parameter:
    """
    Represents a parameter within the PyBOP framework.

    This class encapsulates the definition of a parameter, including its name, prior
    distribution, initial value, bounds, and a margin to ensure the parameter stays
    within feasible limits during optimization or sampling.

    Parameters
    ----------
    name : str
        The name of the parameter.
    initial_value : float, optional
        The initial value to be assigned to the parameter. Defaults to None.
    prior : scipy.stats distribution, optional
        The prior distribution from which parameter values are drawn. Defaults to None.
    bounds : tuple, optional
        A tuple defining the lower and upper bounds for the parameter.
        Defaults to None.

    Raises
    ------
    ValueError
        If the lower bound is not strictly less than the upper bound, or if
        the margin is set outside the interval (0, 1).
    """

    def __init__(
        self, name, initial_value=None, true_value=None, prior=None, bounds=None
    ):
        """
        Construct the parameter class with a name, initial value, prior, and bounds.
        """
        self.name = name
        self.prior = prior
        self.true_value = true_value
        self.initial_value = initial_value
        self.value = initial_value
        self.set_bounds(bounds)
        self.margin = 1e-4

    def rvs(self, n_samples, random_state=None):
        """
        Draw random samples from the parameter's prior distribution.

        The samples are constrained to be within the parameter's bounds, excluding
        a predefined margin at the boundaries.

        Parameters
        ----------
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        array-like
            An array of samples drawn from the prior distribution within the parameter's bounds.
        """
        samples = self.prior.rvs(n_samples, random_state=random_state)

        # Constrain samples to be within bounds
        if self.bounds is not None:
            offset = self.margin * (self.upper_bound - self.lower_bound)
            samples = np.clip(
                samples, self.lower_bound + offset, self.upper_bound - offset
            )

        return samples

    def update(self, value=None, initial_value=None):
        """
        Update the parameter's current value.

        Parameters
        ----------
        value : float
            The new value to be assigned to the parameter.
        """
        if value is not None:
            self.value = value
        elif initial_value is not None:
            self.initial_value = initial_value
            self.value = initial_value
        else:
            raise ValueError("No value provided to update parameter")

    def __repr__(self):
        """
        Return a string representation of the Parameter instance.

        Returns
        -------
        str
            A string including the parameter's name, prior, bounds, and current value.
        """
        return f"Parameter: {self.name} \n Prior: {self.prior} \n Bounds: {self.bounds} \n Value: {self.value}"

    def set_margin(self, margin):
        """
        Set the margin to a specified positive value less than 1.

        The margin is used to ensure parameter samples are not drawn exactly at the bounds,
        which may be problematic in some optimization or sampling algorithms.

        Parameters
        ----------
        margin : float
            The new margin value to be used, which must be in the interval (0, 1).

        Raises
        ------
        ValueError
            If the margin is not between 0 and 1.
        """
        if not 0 < margin < 1:
            raise ValueError("Margin must be between 0 and 1")

        self.margin = margin

    def set_bounds(self, bounds=None):
        """
        Set the upper and lower bounds.

        Parameters
        ----------
        bounds : tuple, optional
            A tuple defining the lower and upper bounds for the parameter.
            Defaults to None.

        Raises
        ------
        ValueError
            If the lower bound is not strictly less than the upper bound, or if
            the margin is set outside the interval (0, 1).
        """
        if bounds is not None:
            if bounds[0] >= bounds[1]:
                raise ValueError("Lower bound must be less than upper bound")
            else:
                self.lower_bound = bounds[0]
                self.upper_bound = bounds[1]

        self.bounds = bounds


class Parameters:
    """
    Represents a set of uncertain parameters within the PyBOP framework.

    This class encapsulates the definition of a parameter, including its name, prior
    distribution, initial value, bounds, and a margin to ensure the parameter stays
    within feasible limits during optimisation or sampling.

    Parameters
    ----------
    parameter_list : pybop.Parameter or Dict
    """

    def __init__(self, *args):
        self.param = OrderedDict()
        for param in args:
            self.add(param)

    def __getitem__(self, key: str):
        """
        Return the parameter dictionary corresponding to a particular key.

        Parameters
        ----------
        key : str
            The name of a parameter.

        Returns
        -------
        pybop.Parameter
            The Parameter object.

        Raises
        ------
        ValueError
            The key must be the name of one of the parameters.
        """
        if key not in self.param.keys():
            raise ValueError(f"The key {key} is not the name of a parameter.")

        return self.param[key]

    def __len__(self) -> int:
        return len(self.param)

    def keys(self) -> List:
        """
        A list of parameter names
        """
        return list(self.param.keys())

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        parameter_names = self.keys()
        if self.index == len(parameter_names):
            raise StopIteration
        name = parameter_names[self.index]
        self.index = self.index + 1
        return self.param[name]

    def add(self, parameter):
        """
        Construct the parameter class with a name, initial value, prior, and bounds.
        """
        if isinstance(parameter, Parameter):
            if parameter.name in self.param.keys():
                raise ValueError(
                    f"There is already a parameter with the name {parameter.name} "
                    + "in the Parameters object. Please remove the duplicate entry."
                )
            self.param[parameter.name] = parameter
        elif isinstance(parameter, dict):
            if "name" not in parameter.keys():
                raise Exception("Parameter requires a name.")
            name = parameter["name"]
            if name in self.param.keys():
                raise ValueError(
                    f"There is already a parameter with the name {name} "
                    + "in the Parameters object. Please remove the duplicate entry."
                )
            self.param[name] = Parameter(**parameter)
        else:
            raise TypeError("Each parameter input must be a Parameter or a dictionary.")

    def remove(self, parameter_name):
        """
        Remove the `Parameter` object from the `Parameters` dictionary.
        """
        if not isinstance(parameter_name, str):
            raise TypeError("The input parameter_name is not a string.")
        if parameter_name not in self.param.keys():
            raise ValueError("This parameter does not exist in the Parameters object.")

        # Remove the parameter
        self.param.pop(parameter_name)

    def get_bounds(self) -> Dict:
        """
        Get bounds, for either all or no parameters.
        """
        all_unbounded = True  # assumption
        bounds = {"lower": [], "upper": []}

        for param in self.param.values():
            if param.bounds is not None:
                bounds["lower"].append(param.bounds[0])
                bounds["upper"].append(param.bounds[1])
                all_unbounded = False
            else:
                bounds["lower"].append(-np.inf)
                bounds["upper"].append(np.inf)
        if all_unbounded:
            bounds = None

        return bounds

    def update(self, values):
        """
        Set value of each parameter.
        """
        for i, param in enumerate(self.param.values()):
            param.update(value=values[i])

    def rvs(self, n_samples: int) -> List:
        """
        Draw random samples from each parameter's prior distribution.

        The samples are constrained to be within the parameter's bounds, excluding
        a predefined margin at the boundaries.

        Parameters
        ----------
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        array-like
            An array of samples drawn from the prior distribution within each parameter's bounds.
        """
        all_samples = []

        for param in self.param.values():
            samples = param.rvs(n_samples)

            # Constrain samples to be within bounds
            if param.bounds is not None:
                offset = param.margin * (param.upper_bound - param.lower_bound)
                samples = np.clip(
                    samples, param.lower_bound + offset, param.upper_bound - offset
                )

            all_samples.append(samples)

        return all_samples

    def get_sigma0(self) -> List:
        """
        Get the standard deviation, for either all or no parameters.
        """
        all_have_sigma = True  # assumption
        sigma0 = []

        for param in self.param.values():
            if hasattr(param.prior, "sigma"):
                sigma0.append(param.prior.sigma)
            else:
                all_have_sigma = False
        if not all_have_sigma:
            sigma0 = None

        return sigma0

    def priors(self) -> List:
        """
        Return the prior distribution of each parameter.
        """
        return [param.prior for param in self.param.values()]

    def initial_value(self) -> List:
        """
        Return the initial value of each parameter.
        """
        initial_values = []

        for param in self.param.values():
            if param.initial_value is None:
                initial_value = param.rvs(1)
                param.update(initial_value=initial_value[0])
            initial_values.append(param.initial_value)

        return initial_values

    def current_value(self) -> List:
        """
        Return the current value of each parameter.
        """
        current_values = []

        for param in self.param.values():
            current_values.append(param.value)

        return current_values

    def true_value(self) -> List:
        """
        Return the true value of each parameter.
        """
        true_values = []

        for param in self.param.values():
            true_values.append(param.true_value)

        return true_values

    def get_bounds_for_plotly(self):
        """
        Retrieve parameter bounds in the format expected by Plotly.

        Returns
        -------
        bounds : numpy.ndarray
            An array of shape (n_parameters, 2) containing the bounds for each parameter.
        """
        bounds = np.empty((len(self), 2))

        for i, param in enumerate(self.param.values()):
            if param.bounds is not None:
                bounds[i] = param.bounds
            else:
                raise ValueError("All parameters require bounds for plotting.")

        return bounds

    def as_dict(self, values=None) -> Dict:
        if values is None:
            values = self.current_value()
        return {key: values[i] for i, key in enumerate(self.param.keys())}
