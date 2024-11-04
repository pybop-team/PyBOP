from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
from pybamm import IDAKLUSolver

from pybop import BaseCost, BaseLikelihood, BaseProblem, Inputs


class BaseJaxCost(BaseCost):
    """
    Jax-based Sum of Squared Error cost function.
    """

    def __init__(self, problem: BaseProblem):
        super().__init__(problem)
        self.model = self.problem.model
        self.n_data = self.problem.n_data
        if isinstance(self.model.solver, IDAKLUSolver):
            self.model.jaxify_solver(t_eval=self.problem.domain_data)

    def __call__(
        self,
        inputs: Inputs,
        calculate_grad: bool = False,
        apply_transform: bool = False,
    ) -> Union[np.array, tuple[float, np.ndarray]]:
        """
        Computes the cost function for the given predictions.

        Parameters
        ----------
        y : dict
            The dictionary of predictions with keys designating the signals for fitting.
        dy : np.ndarray, optional
            The corresponding gradient with respect to the parameters for each signal.
        calculate_grad : bool, optional
            A bool condition designating whether to calculate the gradient.

        Returns
        -------
        float
            The Sum of Squared Error.
        """
        inputs = self.parameters.verify(inputs)
        if calculate_grad != self.model.calculate_sensitivities:
            self._update_solver_sensitivities(calculate_grad)

        if calculate_grad:
            y, dy = jax.value_and_grad(self.evaluate)(inputs)
            return y, np.asarray(
                list(dy.values())
            )  # Convert grad to numpy for optimisers
        else:
            return np.asarray(self.evaluate(inputs))

    def _update_solver_sensitivities(self, calculate_grad: bool) -> None:
        """
        Updates the solver's sensitivity calculation based on the gradient requirement.

        Args:
            calculate_grad (bool): Whether gradient calculation is required.
        """

        self.model.jaxify_solver(
            t_eval=self.problem.domain_data, calculate_sensitivities=calculate_grad
        )

    @staticmethod
    def check_sigma0(sigma0):
        if not isinstance(sigma0, (int, float)) or sigma0 <= 0:
            raise ValueError("sigma0 must be a positive number")
        return float(sigma0)

    def observed_fisher(self, inputs: Inputs):
        """
        Compute the observed fisher information matrix (FIM)
        for the given inputs. This is done with the gradient
        as the Hessian is not available.
        """
        _, grad = self.__call__(inputs, calculate_grad=True)
        return jnp.square(grad) / self.n_data


class JaxSumSquaredError(BaseJaxCost):
    """
    Jax-based Sum of Squared Error cost function.
    """

    def __init__(self, problem: BaseProblem):
        super().__init__(problem)

    def evaluate(self, inputs):
        # Calculate residuals and error
        y = self.problem.evaluate(inputs)
        r = jnp.asarray([y[s] - self._target[s] for s in self.signal])
        return jnp.sum(r**2)


class JaxLogNormalLikelihood(BaseJaxCost, BaseLikelihood):
    """
    A Log-Normal Likelihood function. This function represents the
    underlining observed data sampled from a Log-Normal distribution.

    Parameters
    -----------
    problem: BaseProblem
        The problem to fit of type `pybop.BaseProblem`
    sigma0: float, optional
        The variance in the measured data
    """

    def __init__(self, problem: BaseProblem, sigma0=0.02):
        super().__init__(problem)
        self.sigma = self.check_sigma0(sigma0)
        self.sigma2 = jnp.square(self.sigma)
        self._offset = 0.5 * self.n_data * jnp.log(2 * jnp.pi)
        self._target_as_array = jnp.asarray([self._target[s] for s in self.signal])
        self._log_target_sum = jnp.sum(jnp.log(self._target_as_array))
        self._precompute()

    def _precompute(self):
        self._constant_term = (
            -self._offset - self.n_data * jnp.log(self.sigma) - self._log_target_sum
        )

    def evaluate(self, inputs):
        """
        Evaluates the log-normal likelihood.
        """
        y = self.problem.evaluate(inputs)
        e = jnp.asarray([jnp.log(y[s]) - jnp.log(self._target[s]) for s in self.signal])
        likelihood = self._constant_term - jnp.sum(jnp.square(e)) / (2 * self.sigma2)
        return likelihood


class JaxGaussianLogLikelihoodKnownSigma(BaseJaxCost, BaseLikelihood):
    """
    A Jax implementation of the Gaussian Likelihood function.
    This function represents the underlining observed data sampled
    from a Gaussian distribution with known noise, `sigma0`.

    Parameters
    -----------
    problem: BaseProblem
        The problem to fit of type `pybop.BaseProblem`
    sigma0: float, optional
        The variance in the measured data
    """

    def __init__(self, problem: BaseProblem, sigma0=0.02):
        super().__init__(problem)
        self.sigma = self.check_sigma0(sigma0)
        self.sigma2 = jnp.square(self.sigma)
        self._offset = -0.5 * self.n_data * jnp.log(2 * jnp.pi * self.sigma2)
        self._multip = -1 / (2.0 * self.sigma2)

    def evaluate(self, inputs):
        """
        Evaluates the log-normal likelihood.
        """
        y = self.problem.evaluate(inputs)
        e = jnp.asarray([y[s] - self._target[s] for s in self.signal])
        likelihood = jnp.sum(self._offset + self._multip * jnp.sum(jnp.square(e)))
        return likelihood
