from typing import Optional

import numpy as np

from pybop import BaseLikelihood, BasePrior


class AnnealedImportanceSampler:
    """
    This class implements annealed importance sampling of
    the posterior distribution to compute the model evidence
    introduced in [1].

    [1] "Annealed Importance Sampling", Radford M. Neal, 1998, Technical Report
    No. 9805.
    """

    def __init__(
        self,
        log_likelihood: BaseLikelihood,
        log_prior: BasePrior,
        cov0=None,
        num_beta: int = 30,
        chains: Optional[int] = None,
    ):
        self._log_likelihood = log_likelihood
        self._log_prior = log_prior
        self._cov0 = (
            cov0 if cov0 is not None else np.eye(log_likelihood.n_parameters) * 0.1
        )

        # Total number of iterations
        self._chains = (
            chains if chains is not None else log_likelihood.n_parameters * 300
        )

        # Number of beta divisions to consider 0 = beta_n <
        # beta_n-1 < ... < beta_0 = 1
        self.set_num_beta(num_beta)

    @property
    def chains(self) -> int:
        """Returns the total number of iterations."""
        return self._chains

    @chains.setter
    def chains(self, value: int) -> None:
        """Sets the total number of iterations."""
        if not isinstance(value, (int, np.integer)):
            raise TypeError("iterations must be an integer")
        if value <= 0:
            raise ValueError("iterations must be positive")
        self._chains = int(value)

    @property
    def num_beta(self) -> int:
        """Returns the number of beta points"""
        return self._num_beta

    def set_num_beta(self, num_beta: int) -> None:
        """Sets the number of beta point values."""
        if not isinstance(num_beta, (int, np.integer)):
            raise TypeError("num_beta must be an integer")
        if num_beta <= 1:
            raise ValueError("num_beta must be greater than 1")
        self._num_beta = num_beta
        self._beta = np.linspace(0, 1, num_beta)

    def transition_distribution(self, x, j):
        """
        Transition distribution for each beta value [j] - Eqn 3.
        """
        return (1.0 - self._beta[j]) * self._log_prior(x) + self._beta[
            j
        ] * self._log_likelihood(x)

    def run(self) -> tuple[float, float, float]:
        """
        Run the annealed importance sampling algorithm.

        Returns:
            Tuple containing (mean, median, std, variance) of the log weights

        Raises:
            ValueError: If starting position has non-finite log-likelihood
        """
        log_w = np.zeros(self._chains)
        I = np.zeros(self._chains)
        samples = np.zeros(self._num_beta)

        for i in range(self._chains):
            current = self._log_prior.rvs()
            if not np.isfinite(self._log_likelihood(current)):
                raise ValueError("Starting position has non-finite log-likelihood.")

            current_f = self._log_prior(current)

            log_density_current = np.zeros(self._num_beta)
            log_density_current[0] = current_f
            log_density_previous = np.zeros(self._num_beta)
            log_density_previous[0] = current_f

            # Main sampling loop
            for j in range(1, self._num_beta):
                # Compute jth transition with current sample
                log_density_current[j] = self.transition_distribution(current, j)

                # Calculate the previous transition with current sample
                log_density_previous[j] = self.transition_distribution(current, j - 1)

                # Generate new sample from current (eqn.4)
                proposed = np.random.multivariate_normal(current, self._cov0)

                # Evaluate proposed sample
                if np.isfinite(self._log_likelihood(proposed)):
                    proposed_f = self.transition_distribution(proposed, j)

                    # Metropolis acceptance
                    acceptance_log_prob = proposed_f - current_f
                    if np.log(np.random.rand()) < acceptance_log_prob:
                        current = proposed
                        current_f = proposed_f

                samples[j] = current

            # Sum for weights (eqn.24)
            log_w[i] = (
                np.sum(log_density_current - log_density_previous) / self._num_beta
            )

            # Compute integral using weights and samples
            I[i] = np.mean(
                self._log_likelihood(samples)
                * np.exp((log_density_current - log_density_previous) / self._num_beta)
            )

        # Return log weights, integral, samples
        return log_w, I, samples
