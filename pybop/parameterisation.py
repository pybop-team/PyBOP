import pybop
import pybamm
import numpy as np


class Parameterisation:
    """
    Parameterisation class for pybop.
    """

    def __init__(
        self,
        model,
        observations,
        fit_parameters,
        x0=None,
        check_model=True,
        init_soc=None,
        verbose=False,
    ):
        self.model = model
        self.fit_dict = {}
        self.fit_parameters = {o.name: o for o in fit_parameters}
        self.observations = {o.name: o for o in observations}

        # Check that the observations contain time and current
        for name in ["Time [s]", "Current function [A]"]:
            if name not in self.observations:
                raise ValueError(f"expected {name} in list of observations")

        # Set bounds
        self.bounds = dict(
            lower=[self.fit_parameters[p].bounds[0] for p in self.fit_parameters],
            upper=[self.fit_parameters[p].bounds[1] for p in self.fit_parameters],
        )

        # Sample from prior for x0
        if x0 is None:
            self.x0 = np.zeros(len(self.fit_parameters))
            for i, j in enumerate(self.fit_parameters):
                self.x0[i] = self.fit_parameters[j].prior.rvs(1)[
                    0
                ]  # Updt to capture dimensions per parameter

        # Align initial guess with sample from prior
        for i, j in enumerate(self.fit_parameters):
            self.fit_dict[j] = {j: self.x0[i]}

        # Build parameter set and model
        self.model.build_parameter_set(self.observations, self.fit_parameters)
        self.model.build_model(check_model=check_model, init_soc=init_soc)

    def step(self, signal, x, grad, verbose):
        for i, p in enumerate(self.fit_dict):
            self.fit_dict[p] = x[i]

        y_hat = self.model.solver.solve(
            self.model._built_model, self.model.time_data, inputs=self.fit_dict
        )[signal].data

        try:
            res = np.sqrt(
                np.mean((self.observations["Voltage [V]"].data[1] - y_hat) ** 2)
            )
        except:
            raise ValueError(
                "Measurement and modelled data length mismatch, potentially due to reaching a voltage cut-off"
            )

        if verbose:
            print(self.fit_dict)

        return res

    def map(self, x0):
        """
        Maximum a posteriori estimation.
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

    def rmse(self, signal, method=None):
        """
        Calculate the root mean squared error.
        """

        if method == "nlopt":
            optim = pybop.NLoptOptimize(x0=self.x0)
            results = optim.optimise(
                lambda x, grad: self.step(signal, x, grad), self.x0, self.bounds
            )

        else:
            optim = pybop.NLoptOptimize(method)
            results = optim.optimise(self.step, self.x0, self.bounds)
        return results

    def mle(self, method=None):
        """
        Maximum likelihood estimation.
        """
        pass
