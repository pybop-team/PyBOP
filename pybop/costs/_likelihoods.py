from typing import Optional, Union

import numpy as np
import scipy.stats as stats

import pybop
from pybop.costs.base_cost import BaseCost
from pybop.parameters.parameter import Parameter, Parameters
from pybop.parameters.priors import BasePrior, JointLogPrior, Uniform
from pybop.problems.base_problem import BaseProblem


class BaseLikelihood(BaseCost):
    """
    Base class for likelihoods
    """

    def __init__(self, problem: BaseProblem):
        super().__init__(problem)
        self.n_data = problem.n_data


class GaussianLogLikelihoodKnownSigma(BaseLikelihood):
    """
    This class represents a Gaussian Log Likelihood with a known sigma,
    which assumes that the data follows a Gaussian distribution and computes
    the log-likelihood of observed data under this assumption.

    Parameters
    ----------
    sigma0 : scalar or array
        Initial standard deviation around ``x0``. Either a scalar value (one
        standard deviation for all coordinates) or an array with one entry
        per dimension.
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
        dy: np.ndarray = None,
        calculate_grad: bool = False,
    ) -> Union[float, tuple[float, np.ndarray]]:
        """
        Compute the Gaussian log-likelihood for the given parameters with known sigma.

        This method only computes the likelihood, without calling the problem.evaluateS1.
        """
        # Verify we have dy if calculate_grad is True
        self.verify_args(dy, calculate_grad)

        # Early return if the prediction is not verified
        if not self.verify_prediction(y):
            return (-np.inf, -self.grad_fail) if calculate_grad else -np.inf

        # Calculate residuals and error
        r = np.asarray([self._target[signal] - y[signal] for signal in self.signal])
        e = np.sum(self._offset + self._multip * np.sum(np.real(r * np.conj(r))))

        if calculate_grad:
            dl = np.sum((np.sum((r * dy.T), axis=2) / self.sigma2), axis=1)
            return e, dl

        return e

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
    This class represents a Gaussian Log Likelihood, which assumes that the
    data follows a Gaussian distribution and computes the log-likelihood of
    observed data under this assumption.

    This class estimates the standard deviation of the Gaussian distribution
    alongside the parameters of the model.

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

        # Add sigma parameter, join with self.parameters, reapply transformations
        self.sigma = Parameters()
        self._add_sigma_parameters(sigma0)
        self.join_parameters(self.sigma)
        self.transformation = self._parameters.construct_transformation()

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
                    f"Sigma for output {index+1}",
                    initial_value=value,
                    prior=Uniform(1e-3 * value, 1e3 * value),
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
        dy: np.ndarray = None,
        calculate_grad: bool = False,
    ) -> Union[float, tuple[float, np.ndarray]]:
        """
        Compute the Gaussian log-likelihood for the given parameters.

        This method only computes the likelihood, without calling problem.evaluate().

        Returns
        -------
        float
            The log-likelihood value, or -inf if the standard deviations are non-positive.
        """
        # Verify we have dy if calculate_grad is True
        self.verify_args(dy, calculate_grad)
        sigma = self.sigma.current_value()

        if not self.verify_prediction(y):
            return (-np.inf, -self.grad_fail) if calculate_grad else -np.inf

        # Calculate residuals and error
        r = np.asarray([self._target[signal] - y[signal] for signal in self.signal])
        e = np.sum(
            self._logpi
            - self.n_data * np.log(sigma)
            - np.sum(np.real(r * np.conj(r)), axis=1) / (2.0 * sigma**2.0)
        )

        if calculate_grad:
            dl = np.sum((np.sum((r * dy.T), axis=2) / (sigma**2.0)), axis=1)
            dsigma = (
                -self.n_data / sigma + np.sum(r**2.0, axis=1) / (sigma**3.0)
            ) / self._dsigma_scale
            dl = np.concatenate((dl.flatten(), dsigma))
            return e, dl

        return e


class LogPosterior(BaseLikelihood):
    """
    The Log Posterior for a given problem.

    Computes the log posterior which is proportional to the sum of the log
    likelihood and the log prior.

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
        super().__init__(problem=log_likelihood.problem)
        self.gradient_step = gradient_step

        # Store the likelihood, prior, update parameters and transformation
        self.join_parameters(log_likelihood.parameters)
        self._log_likelihood = log_likelihood

        for attr in ["transformation", "_has_separable_problem"]:
            setattr(self, attr, getattr(log_likelihood, attr))

        if log_prior is None:
            self._prior = JointLogPrior(*self._parameters.priors())
        else:
            self._prior = log_prior

    def compute(
        self,
        y: dict,
        dy: np.ndarray = None,
        calculate_grad: bool = False,
    ) -> Union[float, tuple[float, np.ndarray]]:
        """
        Calculate the posterior cost for a given forward model prediction.

        Parameters
        ----------
        y : dict
            The data for which to evaluate the cost.
        dy : np.ndarray, optional
            The correspond sensitivities in the data.
        calculate_grad : bool, optional
            Whether to calculate the gradient of the cost function.

        Returns
        -------
        Union[float, Tuple[float, np.ndarray]]
            The posterior cost, and optionally the gradient.
        """
        # Verify we have dy if calculate_grad is True
        self.verify_args(dy, calculate_grad)

        if calculate_grad:
            if isinstance(self._prior, BasePrior):
                log_prior, dp = self._prior.logpdfS1(self._parameters.current_value())
            else:
                # Compute log prior first
                log_prior = self._prior.logpdf(self._parameters.current_value())

                # Compute a finite difference approximation of the gradient of the log prior
                delta = self._parameters.initial_value() * self.gradient_step
                dp = []

                for parameter, step_size in zip(self._parameters, delta):
                    param_value = parameter.value
                    upper_value = param_value * (1 + step_size)
                    lower_value = param_value * (1 - step_size)

                    log_prior_upper = parameter.prior.logpdf(upper_value)
                    log_prior_lower = parameter.prior.logpdf(lower_value)

                    gradient = (log_prior_upper - log_prior_lower) / (
                        2 * step_size * param_value + np.finfo(float).eps
                    )
                    dp.append(gradient)
        else:
            log_prior = self._prior.logpdf(self._parameters.current_value())

        if not np.isfinite(log_prior).any():
            return (-np.inf, -self.grad_fail) if calculate_grad else -np.inf

        if calculate_grad:
            log_likelihood, dl = self._log_likelihood.compute(
                y, dy, calculate_grad=True
            )

            posterior = log_likelihood + log_prior
            total_gradient = dl + dp

            return posterior, total_gradient

        log_likelihood = self._log_likelihood.compute(y)
        posterior = log_likelihood + log_prior
        return posterior

    @property
    def prior(self) -> BasePrior:
        return self._prior

    @property
    def likelihood(self) -> BaseLikelihood:
        return self._log_likelihood
