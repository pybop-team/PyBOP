from typing import Optional, Union

import numpy as np
import scipy.stats as stats

import pybop
from pybop.costs.base_cost import BaseCost
from pybop.parameters.parameter import Inputs, Parameter, Parameters
from pybop.parameters.priors import BasePrior, JointLogPrior, Uniform
from pybop.problems.base_problem import BaseProblem


class BaseLikelihood(BaseCost):
    """
    Base class for likelihoods.

    """

    def __init__(self, problem: BaseProblem):
        super().__init__(problem)
        self.n_data = problem.n_data
        self.minimising = False

    def observed_fisher(
        self, inputs: Union[Inputs, list, np.ndarray]
    ) -> Union[np.ndarray, None]:
        """
        Compute the observed Fisher Information Matrix (FIM) for the given data.

        The Fisher information is computed as the outer product of the gradients with respect to
        the model parameters, scaled by the inverse of the dataset size. This method should only
        be used with exponential-based likelihood functions.

        Parameters
        ----------
        inputs : Union[dict[str, float], list-like]
            Input data for model evaluation.

        Returns
        -------
        np.ndarray
            The observed Fisher Information Matrix.
        """

        # Check gradients are available, return None if not.
        if not self.problem.sensitivities_available:
            return

        # Calculate the fisher information via gradient outer-product
        _, grad = self.__call__(inputs, calculate_grad=True)
        shaped_grad = grad.reshape(-1, 1)
        fisher_info = (shaped_grad @ shaped_grad.T) / self.n_data

        return fisher_info


class BaseMetaLikelihood(BaseLikelihood):
    """
    Base class for likelihood classes which have a meta-likelihood such as `LogPosterior` or
    `ScaledLoglikelihood`. This class points the required attributes towards the composed
    likelihood class.
    """

    def __init__(self, log_likelihood: BaseLikelihood):
        self._log_likelihood = log_likelihood
        super().__init__(log_likelihood.problem)

    @property
    def has_separable_problem(self):
        return self._log_likelihood.has_separable_problem

    @property
    def parameters(self):
        return self._log_likelihood.parameters

    @property
    def n_parameters(self):
        return self._log_likelihood.n_parameters

    @property
    def likelihood(self) -> BaseLikelihood:
        return self._log_likelihood


class GaussianLogLikelihoodKnownSigma(BaseLikelihood):
    """
    This class represents a Gaussian log-likelihood with a known sigma, which computes the
    log-likelihood under the assumption that measurement noise on the target data follows a
    Gaussian distribution.

    Parameters
    ----------
    sigma0 : scalar or array
        Initial standard deviation around ``x0``. Either a scalar value (one standard deviation
        for all coordinates) or an array with one entry per dimension.
    """

    def __init__(self, problem: BaseProblem, sigma0: Union[list[float], float]):
        super().__init__(problem)
        sigma0 = self.check_sigma0(sigma0)
        self.sigma2 = sigma0**2.0
        self._offset = -0.5 * self.n_data * np.log(2 * np.pi * self.sigma2)
        self._multip = -1 / (2.0 * self.sigma2)

    def compute(
        self,
        y: dict,
        dy: Optional[np.ndarray] = None,
    ) -> Union[float, tuple[float, np.ndarray]]:
        """
        Compute the Gaussian log-likelihood for the given parameters with known sigma.

        This method only computes the likelihood, without calling the problem.evaluateS1.
        """
        # Early return if the prediction is not verified
        if not self.verify_prediction(y):
            return (-np.inf, -self.grad_fail) if dy is not None else -np.inf

        # Calculate residuals and error
        r = np.asarray([self._target[signal] - y[signal] for signal in self.signal])
        l = np.sum(self._offset + self._multip * np.sum(np.real(r * np.conj(r))))

        if dy is not None:
            dy = self.stack_sensitivities(dy)
            dl = np.sum((np.sum((r * dy), axis=2) / self.sigma2), axis=1)
            return l, dl

        return l

    def check_sigma0(self, sigma0: Union[np.ndarray, float]):
        """
        Check the validity of sigma0.
        """
        sigma0 = np.asarray(sigma0, dtype=float)
        if not np.all(sigma0 > 0):
            raise ValueError("Sigma0 must be positive")
        if np.shape(sigma0) not in [(), (1,), (self.n_outputs,)]:
            raise ValueError(
                "sigma0 must be either a scalar value (one standard deviation for "
                "all coordinates) or an array with one entry per dimension."
            )
        return sigma0


class GaussianLogLikelihood(BaseLikelihood):
    """
    This class represents a Gaussian log-likelihood, which computes the log-likelihood under
    the assumption that measurement noise on the target data follows a Gaussian distribution.

    This class estimates the standard deviation of the Gaussian distribution alongside the
    parameters of the model.

    Attributes
    ----------
    _logpi : float
        Precomputed offset value for the log-likelihood function.
    _dsigma_scale : float
        Scale factor for derivative of standard deviation.
    """

    def __init__(
        self,
        problem: BaseProblem,
        sigma0: Union[float, list[float], list[Parameter]] = 1e-2,
        dsigma_scale: float = 1.0,
    ):
        super().__init__(problem)
        self._dsigma_scale = dsigma_scale
        self._logpi = -0.5 * self.n_data * np.log(2 * np.pi)

        # Add sigma parameter, join with self.parameters
        self.sigma = Parameters()
        self._add_sigma_parameters(sigma0)
        self.join_parameters(self.sigma)

    def _add_sigma_parameters(self, sigma0):
        sigma0 = [sigma0] if not isinstance(sigma0, list) else sigma0
        sigma0 = self._pad_sigma0(sigma0)

        for i, value in enumerate(sigma0):
            self._add_single_sigma(i, value)

    def _pad_sigma0(self, sigma0):
        if len(sigma0) < self.n_outputs:
            return np.pad(
                sigma0,
                (0, self.n_outputs - len(sigma0)),
                constant_values=sigma0[-1],
            )
        return sigma0

    def _add_single_sigma(self, index, value):
        if isinstance(value, Parameter):
            self.sigma.add(value)
        elif isinstance(value, (int, float)):
            self.sigma.add(
                Parameter(
                    f"Sigma for output {index + 1}",
                    initial_value=value,
                    prior=Uniform(1e-8 * value, 3 * value),
                    bounds=[1e-8, 3 * value],
                )
            )
        else:
            raise TypeError(
                f"Expected sigma0 to contain Parameter objects or numeric values. "
                f"Received {type(value)}"
            )

    @property
    def dsigma_scale(self):
        """
        Scaling factor for the dsigma term in the gradient calculation.
        """
        return self._dsigma_scale

    @dsigma_scale.setter
    def dsigma_scale(self, new_value):
        if new_value < 0:
            raise ValueError("dsigma_scale must be non-negative")
        self._dsigma_scale = new_value

    def compute(
        self,
        y: dict,
        dy: Optional[np.ndarray] = None,
    ) -> Union[float, tuple[float, np.ndarray]]:
        """
        Compute the Gaussian log-likelihood for the given parameters.

        This method only computes the likelihood, without calling problem.evaluate().

        Parameters
        ----------
        y : dict[str, np.ndarray[np.float64]]
            The dictionary of predictions with keys designating the signals for fitting.
        dy : dict[str, dict[str, np.ndarray]], optional
            The corresponding gradient with respect to each parameter for each signal.

        Returns
        -------
        np.float64 or tuple[np.float64, np.ndarray[np.float64]]
            If dy is not None, returns a tuple containing the log-likelihood (float) and the
            gradient with dimension (len(parameters)), otherwise returns only the log-likelihood.
        """
        sigma = self.sigma.current_value()

        if not self.verify_prediction(y):
            return (-np.inf, -self.grad_fail) if dy is not None else -np.inf

        # Calculate residuals and error
        r = np.asarray([self._target[signal] - y[signal] for signal in self.signal])
        sum_r2 = np.sum(np.real(r * np.conj(r)), axis=1)
        l = np.sum(
            self._logpi - self.n_data * np.log(sigma) - sum_r2 / (2.0 * sigma**2.0)
        )

        if dy is not None:
            dy = self.stack_sensitivities(dy)
            dl = np.concatenate(
                (
                    np.sum((np.sum((r * dy), axis=2) / (sigma**2.0)), axis=1),
                    (-self.n_data / sigma + sum_r2 / (sigma**3.0)) / self._dsigma_scale,
                )
            )
            return l, dl

        return l


class ScaledLogLikelihood(BaseMetaLikelihood):
    r"""
    This class scaled a `BaseLogLikelihood` class by the number of observations.
    The scaling factor is given below:

    .. math::
       \mathcal{\hat{L(\theta)}} = \frac{1}{N} \mathcal{L(\theta)}

    This class aims to provide numerical values with lower magnitude than the
    canonical likelihoods, which can improve optimiser convergence in certain
    cases.
    """

    def compute(
        self,
        y: dict,
        dy: Optional[np.ndarray] = None,
    ) -> Union[float, tuple[float, np.ndarray]]:
        likelihood = self._log_likelihood.compute(y, dy)
        scaling_factor = 1 / self.n_data

        if isinstance(likelihood, tuple):
            return tuple(val * scaling_factor for val in likelihood)

        return likelihood * scaling_factor


class LogPosterior(BaseMetaLikelihood):
    """
    The Log Posterior for a given problem.

    Computes the log-posterior which is proportional to the sum of the log-likelihood and the
    log-prior.

    Parameters
    ----------
    log_likelihood : BaseLikelihood
        The likelihood class of type ``BaseLikelihood``.
    log_prior : Optional, Union[pybop.BasePrior, stats.rv_continuous]
        The prior class of type ``BasePrior`` or ``stats.rv_continuous``.
        If not provided, the prior class will be taken from the parameter priors
        constructed in the `pybop.Parameters` class.
    gradient_step : float, default: 1e-3
        The step size for the finite-difference gradient calculation
        if the ``log_prior`` is not of type ``BasePrior``.
    """

    def __init__(
        self,
        log_likelihood: BaseLikelihood,
        log_prior: Optional[Union[pybop.BasePrior, stats.rv_continuous]] = None,
        gradient_step: float = 1e-3,
    ):
        self.gradient_step = gradient_step
        super().__init__(log_likelihood)

        if log_prior is None:
            self._prior = JointLogPrior(*self.parameters.priors())
        else:
            self._prior = log_prior

        self._prior.parameters = self.parameters

    def compute(
        self,
        y: dict,
        dy: Optional[np.ndarray] = None,
    ) -> Union[float, tuple[float, np.ndarray]]:
        """
        Calculate the posterior cost for a given forward model prediction.

        Parameters
        ----------
        y : dict[str, np.ndarray[np.float64]]
            The dictionary of predictions with keys designating the signals for fitting.
        dy : dict[str, dict[str, np.ndarray]], optional
            The corresponding sensitivities to each parameter for each signal.

        Returns
        -------
        np.float64 or tuple[np.float64, np.ndarray[np.float64]]
            If dy is not None, returns a tuple containing the log-likelihood (float) and the
            gradient with dimension (len(parameters)), otherwise returns only the log-likelihood.
        """
        # Compute log prior (and gradient)
        if dy is not None:
            if isinstance(self._prior, BasePrior):
                log_prior, dp = self._prior.logpdfS1(self.parameters.current_value())
            else:
                # Compute log prior first
                log_prior = self._prior.logpdf(self.parameters.current_value())

                # Compute a finite difference approximation of the gradient of the log prior
                delta = self.parameters.current_value() * self.gradient_step
                dp = []

                for parameter, step_size in zip(self.parameters, delta):
                    param_value = parameter.value
                    upper_value = param_value + step_size
                    lower_value = param_value - step_size

                    log_prior_upper = self._prior.logpdf(upper_value)
                    log_prior_lower = self._prior.logpdf(lower_value)

                    gradient = (log_prior_upper - log_prior_lower) / (
                        2 * step_size + np.finfo(float).eps
                    )
                    dp.append(gradient)
        else:
            log_prior = self._prior.logpdf(self.parameters.current_value())

        if not np.isfinite(log_prior).any():
            return (-np.inf, -self.grad_fail) if dy is not None else -np.inf

        # Compute log likelihood and add log prior (and gradients)
        if dy is not None:
            log_likelihood, dl = self._log_likelihood.compute(y, dy)

            posterior = log_likelihood + log_prior
            total_gradient = dl + dp

            return posterior, total_gradient

        log_likelihood = self._log_likelihood.compute(y)
        posterior = log_likelihood + log_prior
        return posterior

    @property
    def prior(self) -> BasePrior:
        return self._prior
