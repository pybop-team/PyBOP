import copy
import warnings
from typing import Optional, Union

import numpy as np

from pybop import BaseProblem, DesignProblem
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
    _target : array-like
        An array containing the target data to fit.
    n_outputs : int
        The number of outputs in the model.
    _predict : bool
        If False, the problem will be evaluated outside the self.evaluate() method
        before the cost is calculated (default: False).
    """

    def __init__(self, problem: Optional[BaseProblem] = None):
        self.parameters = Parameters()
        self.problem = problem
        self.verbose = False
        self._predict = False
        self.y = None
        self.dy = None
        self.set_fail_gradient()
        if isinstance(self.problem, BaseProblem):
            self._target = self.problem._target
            self.parameters.join(self.problem.parameters)
            self.n_outputs = self.problem.n_outputs
            self.signal = self.problem.signal
            self._predict = True

    @property
    def n_parameters(self):
        return len(self.parameters)

    def __call__(self, inputs: Union[Inputs, list], grad=None):
        """
        Call the evaluate function for a given set of parameters.
        """
        return self.evaluate(inputs, grad)

    def evaluate(self, inputs: Union[Inputs, list], grad=None):
        """
        Call the evaluate function for a given set of parameters.

        Parameters
        ----------
        inputs : Inputs or array-like
            The parameters for which to compute the cost and gradient.
        grad : array-like, optional
            An array to store the gradient of the cost function with respect
            to the parameters.

        Returns
        -------
        float
            The calculated cost function value.

        Raises
        ------
        ValueError
            If an error occurs during the calculation of the cost.
        """
        inputs = self.parameters.verify(inputs)

        try:
            if self._predict:
                self.y = self.problem.evaluate(inputs)

            return self._evaluate(inputs, grad)

        except NotImplementedError as e:
            raise e

        except Exception as e:
            raise ValueError(f"Error in cost calculation: {e}") from e

    def _evaluate(self, inputs: Inputs, grad=None):
        """
        Calculate the cost function value for a given set of parameters.

        This method must be implemented by subclasses.

        Parameters
        ----------
        inputs : Inputs
            The parameters for which to evaluate the cost.
        grad : array-like, optional
            An array to store the gradient of the cost function with respect
            to the parameters.

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

    def evaluateS1(self, inputs: Union[Inputs, list]):
        """
        Call _evaluateS1 for a given set of parameters.

        Parameters
        ----------
        inputs : Inputs or array-like
            The parameters for which to compute the cost and gradient.

        Returns
        -------
        tuple
            A tuple containing the cost and the gradient. The cost is a float,
            and the gradient is an array-like of the same length as `inputs`.

        Raises
        ------
        ValueError
            If an error occurs during the calculation of the cost or gradient.
        """
        inputs = self.parameters.verify(inputs)

        try:
            if self._predict:
                self.y, self.dy = self.problem.evaluateS1(inputs)

            return self._evaluateS1(inputs)

        except NotImplementedError as e:
            raise e

        except Exception as e:
            raise ValueError(f"Error in cost calculation: {e}") from e

    def _evaluateS1(self, inputs: Inputs):
        """
        Compute the cost and its gradient with respect to the parameters.

        Parameters
        ----------
        inputs : Inputs
            The parameters for which to compute the cost and gradient.

        Returns
        -------
        tuple
            A tuple containing the cost and the gradient. The cost is a float,
            and the gradient is an array-like of the same length as `inputs`.

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


class WeightedCost(BaseCost):
    """
    A subclass for constructing a linear combination of cost functions as
    a single weighted cost function.

    Inherits all parameters and attributes from ``BaseCost``.

    Attributes
    ---------------------
    costs : list[pybop.BaseCost]
        A list of PyBOP cost objects.
    weights : list[float]
        A list of values with which to weight the cost values.
    _has_different_problems : bool
        If True, the problem for each cost is evaluated independently during
        each evaluation of the cost (default: False).
    """

    def __init__(self, *costs, weights: Optional[list[float]] = None):
        if not all(isinstance(cost, BaseCost) for cost in costs):
            raise TypeError("All costs must be instances of BaseCost.")
        self.costs = [copy.copy(cost) for cost in costs]
        self._has_different_problems = False
        self.minimising = not any(
            isinstance(cost.problem, DesignProblem) for cost in self.costs
        )
        if len(set(type(cost.problem) for cost in self.costs)) > 1:
            raise TypeError("All problems must be of the same class type.")

        # Check if weights are provided
        if weights is not None:
            try:
                self.weights = np.asarray(weights, dtype=float)
            except ValueError:
                raise ValueError("Weights must be numeric values.") from None

            if self.weights.size != len(self.costs):
                raise ValueError("Number of weights must match number of costs.")
        else:
            self.weights = np.ones(len(self.costs))

        # Check if all costs depend on the same problem
        self._has_different_problems = any(
            hasattr(cost, "problem") and cost.problem is not self.costs[0].problem
            for cost in self.costs[1:]
        )

        if self._has_different_problems:
            super().__init__()
            for cost in self.costs:
                self.parameters.join(cost.parameters)
        else:
            super().__init__(self.costs[0].problem)
            self._predict = False
            for cost in self.costs:
                cost._predict = False

        # Catch UserWarnings as exceptions
        if not self.minimising:
            warnings.filterwarnings("error", category=UserWarning)

    def _evaluate(self, inputs: Inputs, grad=None):
        """
        Calculate the weighted cost for a given set of parameters.

        Parameters
        ----------
        inputs : Inputs
            The parameters for which to compute the cost.
        grad : array-like, optional
            An array to store the gradient of the cost function with respect
            to the parameters.

        Returns
        -------
        float
            The weighted cost value.
        """
        e = np.empty_like(self.costs)

        if not self._predict:
            if self._has_different_problems:
                self.parameters.update(values=list(inputs.values()))
            else:
                try:
                    with warnings.catch_warnings():
                        self.y = self.problem.evaluate(inputs)
                except UserWarning as e:
                    if self.verbose:
                        print(f"Ignoring this sample due to: {e}")
                    return -np.inf

        for i, cost in enumerate(self.costs):
            if not self._has_different_problems:
                cost.y = self.y
            e[i] = cost.evaluate(inputs)

        return np.dot(e, self.weights)

    def _evaluateS1(self, inputs: Inputs):
        """
        Compute the weighted cost and its gradient with respect to the parameters.

        Parameters
        ----------
        inputs : Inputs
            The parameters for which to compute the cost and gradient.

        Returns
        -------
        tuple
            A tuple containing the cost and the gradient. The cost is a float,
            and the gradient is an array-like of the same length as `x`.
        """
        e = np.empty_like(self.costs)
        de = np.empty((len(self.parameters), len(self.costs)))

        if not self._predict:
            if self._has_different_problems:
                self.parameters.update(values=list(inputs.values()))
            else:
                self.y, self.dy = self.problem.evaluateS1(inputs)

        for i, cost in enumerate(self.costs):
            if not self._has_different_problems:
                cost.y, cost.dy = (self.y, self.dy)
            e[i], de[:, i] = cost.evaluateS1(inputs)

        e = np.dot(e, self.weights)
        de = np.dot(de, self.weights)

        return e, de
