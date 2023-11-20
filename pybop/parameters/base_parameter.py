import numpy as np


class Parameter:
    """ ""
    Class for creating parameters in PyBOP.
    """

    def __init__(self, name, value=None, prior=None, bounds=None):
        self.name = name
        self.prior = prior
        self.value = value
        self.bounds = bounds
        self.lower_bound = self.bounds[0]
        self.upper_bound = self.bounds[1]

        if self.lower_bound > self.upper_bound:
            raise ValueError("Lower bound must be less than upper bound")

    def rvs(self, n_samples):
        """
        Returns a random value sample from the prior distribution.
        """
        samples = self.prior.rvs(n_samples)

        # Constrain samples to be within bounds
        samples = np.clip(samples, self.lower_bound, self.upper_bound)

        # Adjust samples that exactly equal bounds
        samples[samples == self.lower_bound] += samples * 0.0001
        samples[samples == self.upper_bound] -= samples * 0.0001

        return samples

    def update(self, value):
        self.value = value

    def __repr__(self):
        return f"Parameter: {self.name} \n Prior: {self.prior} \n Bounds: {self.bounds} \n Value: {self.value}"
