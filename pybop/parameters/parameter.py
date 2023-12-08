import numpy as np


class Parameter:
    """ ""
    Class for creating parameters in PyBOP.
    """

    def __init__(self, name, initial_value=None, prior=None, bounds=None):
        self.name = name
        self.prior = prior
        self.initial_value = initial_value
        self.value = initial_value
        self.bounds = bounds
        self.lower_bound = self.bounds[0]
        self.upper_bound = self.bounds[1]
        self.margin = 1e-4

        if self.lower_bound >= self.upper_bound:
            raise ValueError("Lower bound must be less than upper bound")

    def rvs(self, n_samples):
        """
        Returns a random value sample from the prior distribution.
        """
        samples = self.prior.rvs(n_samples)

        # Constrain samples to be within bounds
        offset = self.margin * (self.upper_bound - self.lower_bound)
        samples = np.clip(samples, self.lower_bound + offset, self.upper_bound - offset)

        return samples

    def update(self, value):
        self.value = value

    def __repr__(self):
        return f"Parameter: {self.name} \n Prior: {self.prior} \n Bounds: {self.bounds} \n Value: {self.value}"

    def set_margin(self, margin):
        """
        Sets the margin for the parameter.
        """
        if not 0 < margin < 1:
            raise ValueError("Margin must be between 0 and 1")

        self.margin = margin
