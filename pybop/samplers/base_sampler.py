from dataclasses import dataclass

import numpy as np
import pints
import scipy

from pybop import plot
from pybop._logging import Logger
from pybop._result import Result
from pybop.problems.problem import Problem


@dataclass
class SamplerOptions:
    """
    Base options for the sampler.

    Attributes
    ----------
    n_chains : int
        The number of chains to concurrently sample from.
    """

    n_chains: int = 1

    def validate(self):
        """
        Validate the options.

        Raises
        ------
        ValueError
            If the options are invalid.
        """
        if self.n_chains < 1:
            raise ValueError("Number of chains must be greater than 0.")


class BaseSampler:
    """
    Base class for Monte Carlo samplers.

    Parameters
    ----------
    log_pdf : pybop.Problem
        The negative unnormalised posterior distribution.
    options : SamplerOptions, optional
        Options for the sampler. If None, default options are used.
    """

    def __init__(
        self,
        log_pdf: Problem,
        options: SamplerOptions | None = None,
    ):
        self._log_pdf = log_pdf
        self._n_parameters = len(self._log_pdf.parameters)
        self._logger = None
        self._options = options or self.default_options()
        self._options.validate()

        # Get initial conditions
        self._mean0 = self._log_pdf.parameters.get_mean(transformed=True) * np.ones(
            [self._options.n_chains, 1]
        )
        self._cov0 = self._log_pdf.parameters.get_covariance(transformed=True)
        self._validate_covariance_matrix()

    def _validate_covariance_matrix(self) -> None:
        """Check or create the initial covariance matrix."""
        if self._cov0 is None:
            self._cov0 = 0.05
        if np.isscalar(self._cov0):
            self._cov0 = np.eye(self._n_parameters) * self._cov0
        else:
            self._cov0 = np.atleast_2d(self._cov0)
        if (np.atleast_1d(self._cov0) < 0).any():
            raise ValueError("Covariance values must be nonnegative.")

    @staticmethod
    def default_options() -> SamplerOptions:
        """Get the default options for the sampler."""
        return SamplerOptions()

    @property
    def mean0(self) -> np.ndarray:
        return self._mean0

    @property
    def cov0(self) -> np.ndarray:
        return self._cov0

    @property
    def log_pdf(self) -> Problem:
        return self._log_pdf

    @property
    def options(self) -> SamplerOptions:
        return self._options

    def run(self) -> "SamplingResult":
        """
        Sample from the posterior distribution.

        Returns:
            np.ndarray: Samples from the posterior distribution.
        """
        raise NotImplementedError

    def set_initial_phase_iterations(self, iterations: int = 250):
        """Set the number of iterations for the initial phase of the sampler."""
        self._initial_phase_iterations = iterations

    def set_max_iterations(self, iterations: int = 500):
        """Set the maximum number of iterations for the sampler."""
        iterations = int(iterations)
        if iterations < 1:
            raise ValueError("Number of iterations must be greater than 0.")

        self._max_iterations = iterations

    def set_warm_up_iterations(self, iterations: int = 250):
        """Set the number of warm up iterations for the sampler."""
        iterations = int(iterations)
        if iterations < 1:
            raise ValueError("Number of iterations must be greater than 0.")

        self._warm_up = iterations

    @property
    def logger(self) -> Logger | None:
        return self._logger


class SamplingResult(Result):
    """
    Stores the result of the sampling.

    Attributes
    ----------
    sampler : pybop.BaseSampler
        The sampler used to generate the results.
    time : float
        The time taken.
    chains : np.ndarray, optional
        An array containing the samples from the posterior distribution, or None.
    method_name : str
        The name of the sampler.
    message : str
        The reason for stopping given by the sampler.
    """

    def __init__(
        self,
        sampler: "BaseSampler",
        time: float,
        chains: np.ndarray,
        method_name: str | None = None,
        message: str | None = None,
    ):
        super().__init__(
            problem=sampler.log_pdf,
            logger=sampler.logger,
            time=time,
            method_name=method_name,
            message=message,
        )
        self.chains = chains
        self.all_samples = np.concatenate(chains, axis=0)
        self.num_parameters = self.chains.shape[2]

    def signif(self, x, p: int):
        """
        Rounds array `x` to `p` significant digits.
        """
        x = np.asarray(x)
        x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
        mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
        return np.round(x * mags) / mags

    def _calculate_statistics(self, fun, attr_name, *args, **kwargs):
        """
        Calculate statistics from callable `fun`.
        """
        stat = fun(self.all_samples, *args, **kwargs)
        if fun is scipy.stats.mode:
            setattr(self, attr_name, stat[0])
        else:
            setattr(self, attr_name, stat)
        return self.signif(stat, self.sig_digits)

    def get_summary_statistics(self, significant_digits: int = 4):
        """
        Calculate summary statistics for the posterior samples.

        Parameters
        ----------
        significant_digits : int
            Number of significant digits to display for summary statistics.

        Returns
        -------
        dict
            Summary statistics including mean, median, standard deviation, and 95% credible interval.
        """
        self.sig_digits = significant_digits
        summary_funs = {
            "mean": np.mean,
            "median": np.median,
            "mode": scipy.stats.mode,
            "max": np.max,
            "min": np.min,
            "std": np.std,
            "ci_lower": lambda x, axis: np.percentile(x, 2.5, axis=axis),
            "ci_upper": lambda x, axis: np.percentile(x, 97.5, axis=axis),
        }

        return {
            key: self._calculate_statistics(func, key, axis=0)
            for key, func in summary_funs.items()
        }

    def plot_trace(self, **kwargs):
        """
        Plot trace plots for the posterior samples.
        """
        return plot.trace(result=self, **kwargs)

    def plot_chains(self, **kwargs):
        """
        Plot posterior distributions for each chain.
        """
        return plot.chains(result=self, **kwargs)

    def plot_posterior(self, **kwargs):
        """
        Plot the summed posterior distribution across chains.
        """
        return plot.posterior(result=self, **kwargs)

    def summary_table(self, **kwargs):
        """
        Display summary statistics in a table.
        """
        return plot.summary_table(result=self, **kwargs)

    def autocorrelation(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the autocorrelation (Pearson correlation coefficient)
        of a numpy array representing samples.
        """
        x = (x - x.mean()) / (x.std() * np.sqrt(len(x)) + np.finfo(float).eps)
        cor = np.correlate(x, x, mode="full")
        return cor[len(x) : -1]

    def _autocorrelate_negative(self, autocorrelation):
        """
        Returns the index of the first negative entry in ``autocorrelation``, or
        ``len(autocorrelation)`` if a negative entry is not found.
        """
        negative_indices = np.where(autocorrelation < 0)[0]
        return (
            negative_indices[0] if negative_indices.size > 0 else len(autocorrelation)
        )

    def effective_sample_size(self, mixed_chains=False):
        """
        Computes the effective sample size (ESS) for each parameter in each chain.

        Parameters
        ----------
        mixed_chains : bool, optional
            If True, the ESS is computed for all samplers mixed into a single chain.
            Defaults to False.

        Returns
        -------
        list
            A list of effective sample sizes for each parameter in each chain,
            or for the mixed chain if `mixed_chains` is True.

        Raises
        ------
        ValueError
            If there are fewer than two samples in the data.
        """
        if self.all_samples.shape[0] < 2:
            raise ValueError("At least two samples must be given.")

        def compute_ess(samples):
            """Helper function to compute the ESS for a single set of samples."""
            ess = []
            for j in range(self.num_parameters):
                rho = self.autocorrelation(samples[:, j])
                T = self._autocorrelate_negative(rho)
                ess.append(len(samples[:, j]) / (1 + 2 * rho[:T].sum()))
            return ess

        if mixed_chains:
            return compute_ess(self.all_samples)

        ess = []
        for chain in self.chains:
            ess.extend(compute_ess(chain))
        return ess

    def rhat(self):
        """
        Computes the Gelman-Rubin statistic which diagnoses MCMC convergence. For well-mixed and
        stationary chains R-hat will be close to one, otherwise it is higher.
        """
        return pints.rhat(self.chains)
