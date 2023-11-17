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
        sample = self.prior.rvs(n_samples)

        if sample < self.lower_bound:
            return self.lower_bound
        elif sample > self.upper_bound:
            return self.upper_bound
        else:
            return sample

    def update(self, value):
        self.value = value

    def __repr__(self):
        return f"Parameter: {self.name} \n Prior: {self.prior} \n Bounds: {self.bounds} \n Value: {self.value}"
