from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
from pybamm import IDAKLUSolver

from pybop import BaseCost, BaseLikelihood, BaseProblem, Inputs


class BaseJaxCost(BaseCost):
    """
    Base JAX cost function.

    This class implements a cost function using JAX for automatic differentiation
    and efficient gradient computation. It is designed to work with problems
    defined in the `BaseProblem` framework and supports gradient computation.

    Attributes
    ----------
    problem : BaseProblem
        The problem object containing the model, data, and relevant configurations.
    model : BaseModel
        The model associated with the problem.
    n_data : int
        The number of data points in the problem.
    """

    def __init__(self, problem: BaseProblem):
        super().__init__(problem)
        self.model = self.problem.model
        self.n_data = self.problem.n_data

        # JAXify solver if the model uses the IDAKLUSolver
        if isinstance(self.model.solver, IDAKLUSolver):
            self.model.jaxify_solver(t_eval=self.problem.domain_data)

    def single_call(
        self,
        inputs: Inputs,
        calculate_grad: bool = False,
    ) -> Union[np.array, tuple[float, np.ndarray]]:
        """
        Compute the JAX cost function and (optionally) its gradient for given inputs.

        Parameters
        ----------
        inputs : Inputs
            Input data for model evaluation.
        calculate_grad : bool, optional
            Whether to calculate and return the gradient.

        Returns
        -------
        Union[np.ndarray, tuple[float, np.ndarray]]
            The computed cost or a tuple of cost and gradient.
        """
        model_inputs = self.parameters.verify(inputs)

        # Update solver sensitivities if needed
        if calculate_grad != self.model.calculate_sensitivities:
            self._update_solver_sensitivities(calculate_grad)

        if calculate_grad:
            y, dy = jax.value_and_grad(self.evaluate)(model_inputs)
            return y, np.asarray(list(dy.values()))

        return self.evaluate(model_inputs)

    def _update_solver_sensitivities(self, calculate_grad: bool) -> None:
        """
        Updates the solver's sensitivity calculation based on the gradient requirement.

        Parameters
        ----------
            calculate_grad: bool
                Whether gradient calculation is required.
        """

        self.model.jaxify_solver(
            t_eval=self.problem.domain_data, calculate_sensitivities=calculate_grad
        )

    @staticmethod
    def check_sigma0(sigma0):
        """Validates the sigma0 parameter."""
        if not isinstance(sigma0, (int, float)) or sigma0 <= 0:
            raise ValueError("sigma0 must be a positive number")
        return float(sigma0)


class JaxSumSquaredError(BaseJaxCost):
    """
    Jax-based Sum of Squared Error cost function.

    Parameters
    ----------
    problem : BaseProblem
        The problem to fit, of type `pybop.BaseProblem`.

    Methods
    -------
    evaluate(inputs)
        Computes the sum of squared errors between predictions and targets.

    """

    def evaluate(self, inputs):
        """
        Evaluates the sum of squared error for the given predictions.
        """
        y = self.problem.evaluate(inputs)
        residuals = jnp.asarray([y[s] - self._target[s] for s in self.signal])
        return jnp.sum(jnp.square(residuals))


class JaxLogNormalLikelihood(BaseJaxCost, BaseLikelihood):
    """
    A Log-Normal Likelihood function. This function represents the
    underlining observed data sampled from a Log-Normal distribution.

    Parameters
    ----------
    problem : BaseProblem
        The problem to fit, of type `pybop.BaseProblem`.
    sigma0 : float, optional (default=0.02)
        The standard deviation of the measured data.
    """

    def __init__(self, problem: BaseProblem, sigma0: Union[list[float], float]):
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
        Computes the log-normal likelihood.
        """
        y = self.problem.evaluate(inputs)
        residuals = jnp.asarray(
            [jnp.log(y[s]) - jnp.log(self._target[s]) for s in self.signal]
        )
        return self._constant_term - jnp.sum(jnp.square(residuals)) / (2 * self.sigma2)


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

    def __init__(self, problem: BaseProblem, sigma0: Union[list[float], float]):
        super().__init__(problem)
        self.sigma = self.check_sigma0(sigma0)
        self.sigma2 = jnp.square(self.sigma)
        self._offset = -0.5 * self.n_data * jnp.log(2 * jnp.pi * self.sigma2)
        self._multip = -1 / (2.0 * self.sigma2)

    def evaluate(self, inputs):
        """
        Computes the Gaussian log-likelihood.
        """
        y = self.problem.evaluate(inputs)
        residuals = jnp.asarray([y[s] - self._target[s] for s in self.signal])
        return self._offset + jnp.sum(self._multip * jnp.sum(jnp.square(residuals)))
