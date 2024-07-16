import warnings
from collections import OrderedDict
from typing import Union

import numpy as np

from pybop._utils import is_numeric

Inputs = dict[str, float]


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
        self.applied_prior_bounds = False
        self.set_bounds(bounds)
        self.margin = 1e-4

    def rvs(self, n_samples: int = 1, random_state=None):
        """
        Draw random samples from the parameter's prior distribution.

        The samples are constrained to be within the parameter's bounds, excluding
        a predefined margin at the boundaries.

        Parameters
        ----------
        n_samples : int
            The number of samples to draw (default: 1).

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

    def update(self, initial_value=None, value=None):
        """
        Update the parameter's current value.

        Parameters
        ----------
        value : float
            The new value to be assigned to the parameter.
        """
        if initial_value is not None:
            self.initial_value = initial_value
            self.value = initial_value
        if value is not None:
            self.value = value
        if initial_value is None and value is None:
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

    def set_bounds(self, bounds=None, boundary_multiplier=6):
        """
        Set the upper and lower bounds.

        Parameters
        ----------
        bounds : tuple, optional
            A tuple defining the lower and upper bounds for the parameter.
            Defaults to None.
        boundary_multiplier : float, optional
            Used to define the bounds when no bounds are passed but the parameter has
            a prior distribution (default: 6).

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
        elif self.prior is not None:
            self.applied_prior_bounds = True
            self.lower_bound = self.prior.mean - boundary_multiplier * self.prior.sigma
            self.upper_bound = self.prior.mean + boundary_multiplier * self.prior.sigma
            bounds = [self.lower_bound, self.upper_bound]
            print("Default bounds applied based on prior distribution.")

        self.bounds = bounds

    def get_initial_value(self) -> float:
        """
        Return the initial value of each parameter.
        """
        if self.initial_value is None:
            sample = self.rvs(1)
            self.update(initial_value=sample[0])

        return self.initial_value


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

    def __getitem__(self, key: str) -> Parameter:
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

    def keys(self) -> list:
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
                    "in the Parameters object. Please remove the duplicate entry."
                )
            self.param[parameter.name] = parameter
        elif isinstance(parameter, dict):
            if "name" not in parameter.keys():
                raise Exception("Parameter requires a name.")
            name = parameter["name"]
            if name in self.param.keys():
                raise ValueError(
                    f"There is already a parameter with the name {name} "
                    "in the Parameters object. Please remove the duplicate entry."
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

    def join(self, parameters=None):
        """
        Join two Parameters objects into the first by copying across each Parameter.

        Parameters
        ----------
        parameters : pybop.Parameters
        """
        for param in parameters:
            if param not in self.param.values():
                self.add(param)
            else:
                print(f"Discarding duplicate {param.name}.")

    def get_bounds(self) -> dict:
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

    def update(self, initial_values=None, values=None, bounds=None):
        """
        Set value of each parameter.
        """
        for i, param in enumerate(self.param.values()):
            if initial_values is not None:
                param.update(initial_value=initial_values[i])
            if values is not None:
                param.update(value=values[i])
            if bounds is not None:
                if isinstance(bounds, dict):
                    param.set_bounds(bounds=[bounds["lower"][i], bounds["upper"][i]])
                else:
                    param.set_bounds(bounds=bounds[i])

    def rvs(self, n_samples: int = 1) -> np.ndarray:
        """
        Draw random samples from each parameter's prior distribution.

        The samples are constrained to be within the parameter's bounds, excluding
        a predefined margin at the boundaries.

        Parameters
        ----------
        n_samples : int
            The number of samples to draw (default: 1).

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

        return np.concatenate(all_samples)

    def get_sigma0(self) -> list:
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

    def priors(self) -> list:
        """
        Return the prior distribution of each parameter.
        """
        return [param.prior for param in self.param.values()]

    def initial_value(self) -> np.ndarray:
        """
        Return the initial value of each parameter.
        """
        initial_values = []

        for param in self.param.values():
            if param.initial_value is None:
                initial_value = param.rvs(1)[0]
                param.update(initial_value=initial_value)
            initial_values.append(param.initial_value)

        return np.asarray(initial_values)

    def current_value(self) -> np.ndarray:
        """
        Return the current value of each parameter.
        """
        current_values = []

        for param in self.param.values():
            current_values.append(param.value)

        return np.asarray(current_values)

    def true_value(self) -> np.ndarray:
        """
        Return the true value of each parameter.
        """
        true_values = []

        for param in self.param.values():
            true_values.append(param.true_value)

        return np.asarray(true_values)

    def get_bounds_for_plotly(self):
        """
        Retrieve parameter bounds in the format expected by Plotly.

        Returns
        -------
        bounds : numpy.ndarray
            An array of shape (n_parameters, 2) containing the bounds for each parameter.
        """
        bounds = np.zeros((len(self), 2))

        for i, param in enumerate(self.param.values()):
            if param.applied_prior_bounds:
                warnings.warn(
                    "Bounds were created from prior distributions. "
                    "Please provide bounds for better plotting results.",
                    UserWarning,
                    stacklevel=2,
                )
            if param.bounds is not None:
                bounds[i] = param.bounds
            else:
                raise ValueError("All parameters require bounds for plotting.")

        return bounds

    def as_dict(self, values=None) -> dict:
        """
        Parameters
        ----------
        values : list or str, optional
            A list of parameter values or one of the strings "initial" or "true" which can be used
            to obtain a dictionary of parameters.

        Returns
        -------
        Inputs
            A parameters dictionary.
        """
        if values is None:
            values = self.current_value()
        elif isinstance(values, str):
            if values == "initial":
                values = self.initial_value()
            elif values == "true":
                values = self.true_value()
        return {key: values[i] for i, key in enumerate(self.param.keys())}

    def verify(self, inputs: Union[Inputs, None] = None):
        """
        Verify that the inputs are an Inputs dictionary or numeric values
        which can be used to construct an Inputs dictionary

        Parameters
        ----------
        inputs : Inputs or numeric
        """
        if inputs is None or isinstance(inputs, dict):
            return inputs
        elif (isinstance(inputs, list) and all(is_numeric(x) for x in inputs)) or all(
            is_numeric(x) for x in list(inputs)
        ):
            return self.as_dict(inputs)
        else:
            raise TypeError(
                f"Inputs must be a dictionary or numeric. Received {type(inputs)}"
            )
