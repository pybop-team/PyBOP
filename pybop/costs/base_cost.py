from typing import Optional, Union

import numpy as np
from numpy import ndarray

from pybop import BaseProblem
from pybop.parameters.parameter import Inputs, Parameters


class BaseCost:
    """
    Base class for defining cost functions.

    This class is intended to be subclassed to create specific cost functions
    for evaluating model predictions against a set of data. The cost function
    quantifies the goodness-of-fit between the model predictions and the
    observed data, with a lower cost value indicating a better fit.

    Parameters
    ----------
    problem : object
        A problem instance containing the data and functions necessary for
        evaluating the cost function.
    target : array-like
        An array containing the target data to fit.
    n_outputs : int
        The number of outputs in the model.
    has_separable_problem : bool
        If True, the problem is separable from the cost function and will be
        evaluated in advance of the call to self.compute() (default: False).
    _de : float
        The gradient of the cost function to use if an error occurs during
        evaluation. Defaults to 1.0.
    """

    def __init__(self, problem: Optional[BaseProblem] = None):
        self._parameters = Parameters()
        self.transformation = None
        self.problem = problem
        self.verbose = False
        self._has_separable_problem = False
        self.y = None
        self.dy = None
        self._de = 1.0
        if isinstance(self.problem, BaseProblem):
            self._target = self.problem.target
            self._parameters.join(self.problem.parameters)
            self.n_outputs = self.problem.n_outputs
            self.signal = self.problem.signal
            self.transformation = self._parameters.construct_transformation()
            self._has_separable_problem = True
            self.grad_fail = None
            self.set_fail_gradient()

    @property
    def n_parameters(self):
        return len(self._parameters)

    @property
    def has_separable_problem(self):
        return self._has_separable_problem

    @property
    def target(self):
        return self._target

    def __call__(
        self,
        inputs: Union[Inputs, list],
        calculate_grad: bool = False,
        apply_transform: bool = False,
    ):
        """
        This method calls the forward model via problem.evaluate(inputs),
        and computes the cost for the given output by calling self.compute().

        Parameters
        ----------
        inputs : Inputs or array-like
            The parameters for which to compute the cost and gradient.
        calculate_grad : bool, optional
            A bool condition designating whether to calculate the gradient.

        Returns
        -------
        float
            The calculated cost function value.

        Raises
        ------
        ValueError
            If an error occurs during the calculation of the cost.
        """
        # Apply transformation if needed
        self.has_transform = self.transformation is not None and apply_transform
        if self.has_transform:
            inputs = self.transformation.to_model(inputs)
        inputs = self._parameters.verify(inputs)
        self._parameters.update(values=list(inputs.values()))

        y, dy = None, None
        if self._has_separable_problem:
            if calculate_grad:
                y, dy = self.problem.evaluateS1(self.problem.parameters.as_dict())
                cost, grad = self.compute(y, dy=dy, calculate_grad=calculate_grad)
                if self.has_transform and np.isfinite(cost):
                    jac = self.transformation.jacobian(inputs)
                    grad = np.matmul(grad, jac)
                return cost, grad

            y = self.problem.evaluate(self.problem.parameters.as_dict())
        return self.compute(y, dy=dy, calculate_grad=calculate_grad)

    def compute(self, y: dict, dy: ndarray, calculate_grad: bool = False):
        """
        Compute the cost and  if `calculate_grad` is True, its gradient with
        respect to the predictions.

        This method only computes the cost, without calling the `problem.evaluate()`.
        This method must be implemented by subclasses.

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
            The calculated cost function value.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the subclass.
        """
        raise NotImplementedError

    def set_fail_gradient(self, de: float = 1.0):
        """
        Set the fail gradient to a specified value.

        The fail gradient is used if an error occurs during the calculation
        of the gradient. This method allows updating the default gradient value.

        Parameters
        ----------
        de : float
            The new fail gradient value to be used.
        """
        if not isinstance(de, float):
            de = float(de)
        self._de = de
        self.grad_fail = self._de * np.ones(self.n_parameters)

    def verify_prediction(self, y):
        """
        Verify that the prediction matches the target data.

        Parameters
        ----------
        y : dict
            The model predictions.

        Returns
        -------
        bool
            True if the prediction matches the target data, otherwise False.
        """
        if any(
            len(y.get(key, [])) != len(self._target.get(key, [])) for key in self.signal
        ):
            return False

        return True

    def verify_args(self, dy: ndarray, calculate_grad: bool):
        if calculate_grad and dy is None:
            raise ValueError(
                "Forward model sensitivities need to be provided alongside `calculate_grad=True` for `cost.compute`."
            )

    def join_parameters(self, parameters):
        """
        Setter for joining parameters. This method sets the fail gradient if the join adds parameters.
        """
        original_n_params = self.n_parameters
        self._parameters.join(parameters)
        if original_n_params != self.n_parameters:
            self.set_fail_gradient()

    @property
    def parameters(self):
        return self._parameters
