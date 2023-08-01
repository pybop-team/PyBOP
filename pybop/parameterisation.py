import pybop
import pybamm
import numpy as np


class Parameterisation:
    """
    Parameterisation class for pybop.
    """

    def __init__(self, model, observations, fit_parameters, x0=None, options=None):
        self.model = model
        self.fit_parameters = fit_parameters
        self.observations = observations
        self.options = options

        # To Do:
        # Split observations into forward model inputs/outputs
        # checks on observations and parameters

        if options is not None:
            self.parameter_set = options.parameter_set
        else:
            self.parameter_set = self.model.pybamm_model.default_parameter_values

        if x0 is None:
            self.x0 = np.ones(len(self.fit_parameters)) * 0.1

        self.sim = pybamm.Simulation(
            self.model.pybamm_model, parameter_values=self.parameter_set
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
            for i in range(len(self.fit_parameters)):
                self.sim.parameter_set.update(
                    {
                        self.fit_parameters[i]
                        .name: self.fit_parameters[i]
                        .prior.rvs(1)[0]
                    }
                )

            y_hat = self.sim.solve()["Terminal voltage [V]"].data
            return np.sqrt(np.mean((self.observations["Voltage [V]"] - y_hat) ** 2))

        if method == "nlopt":
            results = pybop.nlopt_opt(
                step, self.x0, [p.bounds for p in self.fit_parameters], self.options
            )
        else:
            results = pybop.scipy_opt.optimise(
                step, self.x0, [p.bounds for p in self.fit_parameters], self.options
            )
        return results

    def mle(self, method):
        """
        Maximum likelihood estimation.
        """
        pass

        [p for p in self.fit_parameters]
