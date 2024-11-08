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
        x0=None,
        cov0=None,
        num_beta: int = 30,
        iterations: Optional[int] = None,
    ):
        self._log_likelihood = log_likelihood
        self._log_prior = log_prior

        # Set defaults for x0, cov0
        self._x0 = (
            x0 if x0 is not None else log_likelihood.parameters.initial_value()
        )  # Needs transformation
        self._cov0 = (
            cov0 if cov0 is not None else np.eye(log_likelihood.n_parameters) * 0.1
        )

        # Total number of iterations
        self._iterations = (
            iterations if iterations is not None else log_likelihood.n_parameters * 1000
        )

        # Number of beta divisions to consider 0 = beta_n <
        # beta_n-1 < ... < beta_0 = 1
        self._num_beta = 30

        self.set_num_beta(num_beta)

    @property
    def iterations(self) -> int:
        """Returns the total number of iterations."""
        return self._iterations

    @iterations.setter
    def iterations(self, value: int) -> None:
        """Sets the total number of iterations."""
        if not isinstance(value, (int, np.integer)):
            raise TypeError("iterations must be an integer")
        if value <= 0:
            raise ValueError("iterations must be positive")
        self._iterations = int(value)

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

    def run(self) -> tuple[float, float, float, float]:
        """
        Run the annealed importance sampling algorithm.

        Returns:
            Tuple containing (mean, std, median, variance) of the log weights

        Raises:
            ValueError: If starting position has non-finite log-likelihood
        """
        log_w = np.zeros(self._iterations)

        for i in range(self._iterations):
            current = self._x0.copy()

            if not np.isfinite(self._log_likelihood(current)):
                raise ValueError("Starting position has non-finite log-likelihood.")

            log_likelihood_current = self._log_likelihood(current)
            log_prior_current = self._log_prior(current)
            current_f = log_prior_current + self._beta[0] * log_likelihood_current

            log_density_current = np.zeros(self._num_beta)
            log_density_previous = np.zeros(self._num_beta)
            log_density_previous[0] = current_f

            # Main sampling loop
            for j in range(1, self._num_beta):
                proposed = np.random.multivariate_normal(current, self._cov0)

                # Evaluate proposed state
                log_likelihood_proposed = self._log_likelihood(proposed)
                log_prior_proposed = self._log_prior(proposed)

                # Store proposed
                log_density_current[j - 1] = current_f

                # Metropolis sampling step
                if np.isfinite(log_likelihood_proposed):
                    proposed_f = (
                        log_prior_proposed + self._beta[j] * log_likelihood_proposed
                    )
                    acceptance_log_prob = proposed_f - current_f

                    if np.log(np.random.rand()) < acceptance_log_prob:
                        current = proposed
                        current_f = proposed_f

                log_density_previous[j] = current_f

                # Final state
                log_density_current[self._num_beta - 1] = self._log_prior(
                    current
                ) + self._log_likelihood(current)
            log_w[i] = np.sum(log_density_current) - np.sum(log_density_previous)

        # Return moments of generated chain
        return np.mean(log_w), np.median(log_w), np.std(log_w), np.var(log_w)
