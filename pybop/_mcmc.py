import torch
import pyro
import numpy as np
import pyro.distributions as dist
from pyro.infer import MCMC


class MonteCarloSampler:
    """
    Class for Monte Carlo Sampling
    """

    def __init__(self, problem, kernel, parameter_priors):
        self.problem = problem
        self.posterior = None
        self.parameter_priors = parameter_priors
        self.kernel = getattr(pyro.infer, kernel)(self.mcmc_model)

    def mcmc_model(self):
        input_params = np.zeros(len(self.parameter_priors))

        for i, (param_name, (alpha, beta, lower_bound, upper_bound)) in enumerate(
            self.parameter_priors.items()
        ):
            # Sample from the Beta distribution (in the 0, 1 range)
            unconstrained = pyro.sample(
                f"{param_name}_unconstrained", dist.Beta(alpha, beta)
            )
            # Transform to the desired interval, and store the value
            input_params[i] = self.transform_to_interval(
                unconstrained, lower_bound, upper_bound
            ).item()

        # Predict the values from the model
        # predicted_values = model.predict(t_eval=t_eval)
        predicted_values = self.problem.evaluate(x=input_params)

        # Convert your predicted and observed data to PyTorch tensors
        voltage_obs = torch.tensor(
            self.problem._dataset[self.problem.signal].data, dtype=torch.float32
        )
        voltage_pred = torch.tensor(predicted_values, dtype=torch.float32)
        # sigma_tensor = torch.tensor(sigma, dtype=torch.float32)

        # Likelihood of the observations
        pyro.sample(
            "obs", dist.Normal(voltage_pred, 0.001).to_event(1), obs=voltage_obs
        )

    def run(self, num_samples=100, warmup_steps=200, num_chains=1):
        """
        Run the Monte Carlo Sampling
        """

        mcmc = MCMC(
            self.kernel,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            num_chains=num_chains,
        )
        mcmc.run()

        return mcmc.get_samples()

    def transform_to_interval(self, unconstrained, lower_bound, upper_bound):
        return unconstrained * (upper_bound - lower_bound) + lower_bound
