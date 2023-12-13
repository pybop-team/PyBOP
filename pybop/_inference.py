import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC


class BayesianSampler:
    """
    Performs Bayesian inference using Markov Chain Monte Carlo (MCMC) sampling with Pyro.

    Parameters
    ----------
    problem : object
        The problem instance containing the model and dataset to be sampled from.
    kernel : str
        The name of the Pyro sampling kernel to be used for MCMC.
    parameter_priors : dict
        A dictionary of prior distributions for the model parameters.
    """

    def __init__(self, problem, kernel, parameter_priors):
        self.problem = problem
        self.posterior = None
        self.parameter_priors = parameter_priors
        self.kernel = getattr(pyro.infer, kernel)(self.define_sampling_model)
        self.input_params = torch.zeros(len(self.parameter_priors), dtype=torch.float32)

    def define_sampling_model(self):
        """
        Defines the model for MCMC sampling, including the prior and likelihood.
        """

        # Transform the parameters to the desired interval
        self.transform_to_parameter_space(self.parameter_priors)

        # Predict the values from the model
        predicted_values = self.problem.evaluate(x=self.input_params.detach().numpy())

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
        Run the MCMC algorithm to sample from the posterior distribution.

        Parameters
        ----------
        num_samples : int, optional
            The number of samples to draw from the posterior. (default: 100)
        warmup_steps : int, optional
            The number of warm-up steps before sampling begins. (default: 200)
        num_chains : int, optional
            The number of MCMC chains to run. (default: 1)

        Returns
        -------
        dict
            A dictionary of samples from the posterior distribution.
        """

        self.mcmc = MCMC(
            self.kernel,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            num_chains=num_chains,
        )
        self.mcmc.run()
        self.samples = self.mcmc.get_samples()
        self.transform_to_parameter_space(self.parameter_priors, samples=self.samples)

        return self.samples

    def transform_to_parameter_space(self, parameter_priors, samples=None):
        """
        Transforms unconstrained samples to the parameter space defined by the priors.

        Parameters
        ----------
        parameter_priors : dict
            A dictionary containing the parameter names and their corresponding prior distributions.
        samples : dict, optional
            A dictionary containing unconstrained samples to be transformed. If None, new samples are drawn. (default: None)
        """
        if samples is None:
            for i, (param_name, (alpha, beta, lower_bound, upper_bound)) in enumerate(
                parameter_priors.items()
            ):
                # Sample from the Beta distribution (in the 0, 1 range)
                unconstrained = pyro.sample(
                    f"{param_name}_unconstrained", dist.Beta(alpha, beta)
                )

                # Transform to the desired interval, and store the value
                self.input_params[i] = self.scale_to_bounds(
                    unconstrained, lower_bound, upper_bound
                )
        else:
            for i, (param_name, (alpha, beta, lower_bound, upper_bound)) in enumerate(
                parameter_priors.items()
            ):
                # Transform to the desired interval, and store the value
                self.samples.update(
                    {
                        param_name: self.scale_to_bounds(
                            samples[f"{param_name}_unconstrained"],
                            lower_bound,
                            upper_bound,
                        )
                    }
                )

    @staticmethod
    def scale_to_bounds(unconstrained, lower_bound, upper_bound):
        """
        Transforms a sample from the [0, 1] interval to the desired [lower_bound, upper_bound] interval.

        Parameters
        ----------
        unconstrained : float or torch.Tensor
            The value in the [0, 1] interval to be transformed.
        lower_bound : float
            The lower bound of the target interval.
        upper_bound : float
            The upper bound of the target interval.

        Returns
        -------
        float or torch.Tensor
            The transformed value in the [lower_bound, upper_bound] interval.
        """
        return unconstrained * (upper_bound - lower_bound) + lower_bound
