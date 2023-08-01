import pybop
import pybamm
import numpy as np


class Parameterisation:
    """
    Parameterisation class for pybop.
    """

    def __init__(self, model, observations, parameters, x0=None, options=None):
        self.model = model
        self.parameters = parameters
        self.observations = observations
        self.options = options
        self.default_parameters = (
            parameters.default_parameters
            or self.model.pybamm_model.default_parameter_values
        )
        # To Do:
        # Split observations into forward model inputs/outputs
        # checks on observations and parameters
        
        if x0 is None:
            self.x0 = np.zeros(len(self.parameters))

        self.sim = pybop.Simulation(
            self.model.pybamm_model, parameter_values=self.default_parameters
        )

    def map(self, x0):
        """
        Max a posteriori estimation.
        """
        pass

    def sample(self, n_chains):
        """
        Sample from the posterior distribution.
        """
        pass

    def rmse(self, method=None):
        """
        Calculate the root mean squared error.
        """

        def step(x):
            self.parameters.update(lambda x: {p: x[i] for i, p in self.parameters})
            y_hat = self.sim.solve()["Terminal voltage [V]"].data
            return np.sqrt(np.mean((self.observations["Voltage [V]"] - y_hat) ** 2))

        if method == "nlopt":
            results = pybop.opt.nlopt(
                step, self.x0, self.parameters.bounds, self.options
            )
        else:
            results = pybop.opt.scipy(
                step, self.x0, self.parameters.bounds, self.options
            )
        return results

    def mle(self, method):
        """
        Maximum likelihood estimation.
        """
        pass
