import pybop
import pybamm
import numpy as np


# To Do:
# checks on observations and parameters
# implement method to update parameterisation without reconstructing the simulation
# Geometric parameters might always require reconstruction (i.e. electrode height)


class Parameterisation:
    """
    Parameterisation class for pybop.
    """

    def __init__(self, model, observations, fit_parameters, x0=None, options=None):
        self.model = model
        self.fit_parameters = fit_parameters
        self.observations = observations
        self.options = options
        self.bounds = dict(
            lower=[p.bounds[0] for p in fit_parameters],
            upper=[p.bounds[1] for p in fit_parameters],
        )

        if options is not None:
            self.parameter_set = options.parameter_set
        else:
            self.parameter_set = self.model.pybamm_model.default_parameter_values

        try:
            self.parameter_set["Current function [A]"] = pybamm.Interpolant(
                self.observations["Time"].data,
                self.observations["Current"].data,
                pybamm.t,
            )
        except:
            raise ValueError("Current function not supplied")

        if x0 is None:
            self.x0 = np.mean([self.bounds["lower"], self.bounds["upper"]], axis=0)

        self.sim = pybop.Simulation(
            self.model.pybamm_model,
            parameter_values=self.parameter_set,
            solver=pybamm.CasadiSolver(),
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

    def pem(self, method=None):
        """
        Predictive error minimisation.
        """
        pass

    def rmse(self, method=None):
        """
        Calculate the root mean squared error.
        """

        def step(x, grad):
            for i in range(len(self.fit_parameters)):
                self.parameter_set.update({self.fit_parameters[i].name: x[i]})

            self.sim = pybamm.Simulation(
                self.model.pybamm_model,
                parameter_values=self.parameter_set,
                solver=pybamm.CasadiSolver(),
            )
            y_hat = self.sim.solve()["Terminal voltage [V]"].data

            try:
                res = np.sqrt(
                    np.mean((self.observations["Voltage"].data[1] - y_hat) ** 2)
                )
            except:
                raise ValueError(
                    "Measurement and modelled data length mismatch, potentially due to voltage cut-offs"
                )
            return res

        if method == "nlopt":
            optim = pybop.NLoptOptimize(x0=self.x0)
            results = optim.optimise(step, self.x0, self.bounds)
        else:
            optim = pybop.NLoptOptimize(method)
            results = optim.optimise(step, self.x0, self.bounds)
        return results

    def mle(self, method=None):
        """
        Maximum likelihood estimation.
        """
        pass

