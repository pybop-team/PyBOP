from typing import Optional, Union

import numpy as np
from numpy import ndarray

from pybop import BaseProblem
from pybop._utils import add_spaces
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
        self._transformation = None
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
            self._transformation = self._parameters.construct_transformation()
            self._has_separable_problem = True
            self.grad_fail = None
            self.set_fail_gradient()

    def __call__(
        self,
        inputs: Union[Inputs, list],
        calculate_grad: bool = False,
        apply_transform: bool = False,
    ) -> Union[float, tuple[float, np.ndarray]]:
        """
        This method calls the forward model via problem.evaluate(inputs),
        and computes the cost for the given output by calling self.compute().

        Parameters
        ----------
        inputs : Inputs or list-like
            The input parameters for which the cost and optionally the gradient
            will be computed.
        calculate_grad : bool, optional, default=False
            If True, both the cost and gradient will be computed. Otherwise, only the
            cost is computed.
        apply_transform : bool, optional, default=False
            If True, applies a transformation to the inputs before evaluating the model.

        Returns
        -------
        float or tuple
            - If `calculate_grad` is False, returns the computed cost (float).
            - If `calculate_grad` is True, returns a tuple containing the cost (float)
              and the gradient (np.ndarray).

        Raises
        ------
        ValueError
            If an error occurs during the calculation of the cost.
        """
        # Note, we use the transformation and parameter properties here to enable
        # differing attributes within the `LogPosterior` class

        # Apply transformation if needed
        self.has_transform = self.transformation is not None and apply_transform
        if self.has_transform:
            model_inputs = self.transformation.to_model(inputs)
        else:
            model_inputs = inputs

        # Validate inputs, update parameters
        model_inputs = self.parameters.verify(model_inputs)
        self.parameters.update(values=list(model_inputs.values()))

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
    def name(self):
        return add_spaces(type(self).__name__)

    @property
    def n_parameters(self):
        return len(self._parameters)

    @property
    def has_separable_problem(self):
        return self._has_separable_problem

    @property
    def target(self):
        return self._target

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = parameters

    @property
    def transformation(self):
        return self._transformation

    @transformation.setter
    def transformation(self, transformation):
        self._transformation = transformation
