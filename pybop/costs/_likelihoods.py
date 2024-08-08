from typing import Union

import numpy as np

from pybop.costs.base_cost import BaseCost
from pybop.parameters.parameter import Inputs, Parameter, Parameters
from pybop.parameters.priors import Uniform
from pybop.problems.base_problem import BaseProblem


class BaseLikelihood(BaseCost):
    """
    Base class for likelihoods
    """

    def __init__(self, problem: BaseProblem):
        super().__init__(problem)
        self.n_time_data = problem.n_time_data


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
        self._offset = -0.5 * self.n_time_data * np.log(2 * np.pi * self.sigma2)
        self._multip = -1 / (2.0 * self.sigma2)

    def compute(
        self, inputs: Inputs, calculate_grad: bool = False
    ) -> Union[float, tuple[float, np.ndarray]]:
        """
        Compute the Gaussian log-likelihood for the given parameters with known sigma.

        This method only computes the likelihood, without calling the problem.evaluateS1.
        """
        # Early return if the prediction is not verified
        if not self.verify_prediction(self.y):
            return (
                (-np.inf, -self._de * np.ones(self.n_parameters))
                if calculate_grad
                else -np.inf
            )

        # Calculate residuals and error
        r = np.asarray(
            [self._target[signal] - self.y[signal] for signal in self.signal]
        )
        e = np.sum(self._offset + self._multip * np.sum(r**2.0))

        if calculate_grad:
            dl = np.sum((np.sum((r * self.dy.T), axis=2) / self.sigma2), axis=1)
            return e, dl

        return e.item() if self.n_outputs == 1 else np.sum(e)

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
        sigma0: Union[float, list[float], list[Parameter]] = 0.002,
        dsigma_scale: float = 1.0,
    ):
        super().__init__(problem)
        self._dsigma_scale = dsigma_scale
        self._logpi = -0.5 * self.n_time_data * np.log(2 * np.pi)
        self._has_separable_problem = False

        self.sigma = Parameters()
        self._add_sigma_parameters(sigma0)
        self.parameters.join(self.sigma)

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
                    prior=Uniform(0.5 * value, 1.5 * value),
                    # bounds = [1e-6, 1.0], # TODO: This should have a lower bound of zero and no higher bound
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
        self, inputs: Inputs, calculate_grad: bool = False
    ) -> Union[float, tuple[float, np.ndarray]]:
        """
        Compute the Gaussian log-likelihood for the given parameters.

        This method only computes the likelihood, without calling the problem.evaluateS1.

        Parameters
        ----------
        inputs : Inputs
            The parameters for which to evaluate the log-likelihood, including the `n_outputs`
            standard deviations of the Gaussian distributions.

        Returns
        -------
        float
            The log-likelihood value, or -inf if the standard deviations are non-positive.
        """
        self.parameters.update(values=list(inputs.values()))
        sigma = self.sigma.current_value()
        if np.any(sigma <= 0):
            return (
                (-np.inf, -self._de * np.ones(self.n_parameters))
                if calculate_grad
                else -np.inf
            )

        if calculate_grad:
            self.y, self.dy = self.problem.evaluateS1(self.problem.parameters.as_dict())
        else:
            self.y = self.problem.evaluate(self.problem.parameters.as_dict())

        if not self.verify_prediction(self.y):
            return (
                (-np.inf, -self._de * np.ones(self.n_parameters))
                if calculate_grad
                else -np.inf
            )

        # Calculate residuals and error
        r = np.asarray(
            [self._target[signal] - self.y[signal] for signal in self.signal]
        )
        e = np.sum(
            self._logpi
            - self.n_time_data * np.log(sigma)
            - np.sum((r) ** 2.0) / (2.0 * sigma**2.0)
        )

        if calculate_grad:
            dl = np.sum((np.sum((r * self.dy.T), axis=2) / (sigma**2.0)), axis=1)
            dsigma = (
                -self.n_time_data / sigma + np.sum(r**2.0, axis=1) / (sigma**3.0)
            ) / self._dsigma_scale
            dl = np.concatenate((dl.flatten(), dsigma))
            return (e.item(), dl) if self.n_outputs == 1 else (np.sum(e), dl)

        return e.item() if self.n_outputs == 1 else np.sum(e)


class MAP(BaseLikelihood):
    """
    Maximum a posteriori cost function.

    Computes the maximum a posteriori cost function, which is the sum of the
    log likelihood and the log prior. The goal of maximising is achieved by
    setting minimising = False in the optimiser settings.

    Inherits all parameters and attributes from ``BaseLikelihood``.

    """

    def __init__(self, problem, likelihood, sigma0=None, gradient_step=1e-3):
        super().__init__(problem)
        self.sigma0 = sigma0
        self.gradient_step = gradient_step
        if self.sigma0 is None:
            self.sigma0 = []
            for param in self.problem.parameters:
                self.sigma0.append(param.prior.sigma)

        try:
            self.likelihood = likelihood(problem=self.problem, sigma0=self.sigma0)
        except Exception as e:
            raise ValueError(
                f"An error occurred when constructing the Likelihood class: {e}"
            ) from e

        if hasattr(self, "likelihood") and not isinstance(
            self.likelihood, BaseLikelihood
        ):
            raise ValueError(f"{self.likelihood} must be a subclass of BaseLikelihood")

        self.parameters = self.likelihood.parameters
        self._has_separable_problem = self.likelihood.has_separable_problem

    def compute(
        self, inputs: Inputs, calculate_grad: bool = False
    ) -> Union[float, tuple[float, np.ndarray]]:
        """
        Compute the Maximum a Posteriori for the given parameters.

        If self._has_separable_problem is True, then this method only computes the
        likelihood, without calling the problem.evaluate(). Else, problem.evaluate()
        is called before computing the likelihood.

        Parameters
        ----------
        inputs : Inputs
            The parameters for which to compute the cost.

        Returns
        -------
        float
            The maximum a posteriori cost.
        """
        log_prior = sum(
            self.parameters[key].prior.logpdf(value) for key, value in inputs.items()
        )

        if not np.isfinite(log_prior).any():
            return (
                (-np.inf, -self._de * np.ones(self.n_parameters))
                if calculate_grad
                else -np.inf
            )

        if self._has_separable_problem:
            self.likelihood.y = self.y
            if calculate_grad:
                self.likelihood.dy = self.dy
        if calculate_grad:
            log_likelihood, dl = self.likelihood(inputs, calculate_grad=True)

            # Compute a finite difference approximation of the gradient of the log prior
            delta = self.parameters.initial_value() * self.gradient_step
            prior_gradient = []

            for parameter, step_size in zip(self.problem.parameters, delta):
                param_value = inputs[parameter.name]

                log_prior_upper = parameter.prior.logpdf(param_value * (1 + step_size))
                log_prior_lower = parameter.prior.logpdf(param_value * (1 - step_size))

                gradient = (log_prior_upper - log_prior_lower) / (
                    2 * step_size * param_value + np.finfo(float).eps
                )
                prior_gradient.append(gradient)

            posterior = log_likelihood + log_prior
            total_gradient = dl + prior_gradient

            return posterior, total_gradient

        log_likelihood = self.likelihood(inputs)
        posterior = log_likelihood + log_prior
        return posterior
