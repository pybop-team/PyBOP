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
