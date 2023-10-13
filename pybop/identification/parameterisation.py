import pybop
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
        self.verbose = verbose
        self.fit_dict = {}
        self.fit_parameters = {o.name: o for o in fit_parameters}
        self.observations = {o.name: o for o in observations}
        self.model.n_parameters = len(self.fit_dict)

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

        # Build model with observations and fitting_parameters
        self.model.build(
            self.observations,
            self.fit_parameters,
            check_model=check_model,
            init_soc=init_soc,
        )

    def step(self, signal, x, grad):
        for i, p in enumerate(self.fit_dict):
            self.fit_dict[p] = x[i]

        y_hat = self.model.solver.solve(
            self.model.built_model, self.model.time_data, inputs=self.fit_dict
        )[signal].data

        print(
            "Last Voltage Values:", y_hat[-1], self.observations["Voltage [V]"].data[-1]
        )

        if len(y_hat) != len(self.observations["Voltage [V]"].data):
            print(
                "len of vectors:",
                len(y_hat),
                len(self.observations["Voltage [V]"].data),
            )
            raise ValueError(
                "Measurement and simulated data length mismatch, potentially due to reaching a voltage cut-off"
            )

        try:
            res = np.sqrt(np.mean((self.observations["Voltage [V]"].data - y_hat) ** 2))
        except:
            print("Error in RMSE calculation")

        if self.verbose:
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
