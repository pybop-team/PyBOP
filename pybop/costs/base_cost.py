import numpy as np

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
    _target : array-like
        An array containing the target data to fit.
    n_outputs : int
        The number of outputs in the model.
    """

    def __init__(self, problem=None):
        self.parameters = Parameters()
        self.problem = problem
        self._fixed_problem = False
        if isinstance(self.problem, BaseProblem):
            self._target = self.problem._target
            self.parameters = self.problem.parameters
            self.n_outputs = self.problem.n_outputs
            self.signal = self.problem.signal
            self._fixed_problem = True

    @property
    def n_parameters(self):
        return len(self.parameters)

    def __call__(self, x, grad=None):
        """
        Call the evaluate function for a given set of parameters.
        """
        return self.evaluate(x, grad)

    def evaluate(self, x, grad=None):
        """
        Call the evaluate function for a given set of parameters.

        Parameters
        ----------
        x : array-like
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
        ValueError
            If an error occurs during the calculation of the cost.
        """
        inputs = self.parameters.verify(x)

        try:
            if self._fixed_problem:
                self._current_prediction = self.problem.evaluate(inputs)

            return self._evaluate(inputs, grad)

        except NotImplementedError as e:
            raise e

        except Exception as e:
            raise ValueError(f"Error in cost calculation: {e}")

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

    def evaluateS1(self, x):
        """
        Call _evaluateS1 for a given set of parameters.

        Parameters
        ----------
        x : array-like
            The parameters for which to compute the cost and gradient.

        Returns
        -------
        tuple
            A tuple containing the cost and the gradient. The cost is a float,
            and the gradient is an array-like of the same length as `x`.

        Raises
        ------
        ValueError
            If an error occurs during the calculation of the cost or gradient.
        """
        inputs = self.parameters.verify(x)

        try:
            if self._fixed_problem:
                self._current_prediction, self._current_sensitivities = (
                    self.problem.evaluateS1(inputs)
                )

            return self._evaluateS1(inputs)

        except NotImplementedError as e:
            raise e

        except Exception as e:
            raise ValueError(f"Error in cost calculation: {e}")

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
            and the gradient is an array-like of the same length as `x`.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the subclass.
        """
        raise NotImplementedError


class WeightedCost(BaseCost):
    """
    A subclass for constructing a linear combination of cost functions as
    a single weighted cost function.

    Inherits all parameters and attributes from ``BaseCost``.
    """

    def __init__(self, cost_list, weights=None):
        self.cost_list = cost_list
        self.weights = weights
        self._different_problems = False

        if not isinstance(self.cost_list, list):
            raise TypeError(
                f"Expected a list of costs. Received {type(self.cost_list)}"
            )
        if self.weights is None:
            self.weights = np.ones(len(cost_list))
        elif isinstance(self.weights, list):
            self.weights = np.array(self.weights)
        if not isinstance(self.weights, np.ndarray):
            raise TypeError(
                "Expected a list or array of weights the same length as cost_list."
            )
        if not len(self.weights) == len(self.cost_list):
            raise ValueError(
                "Expected a list or array of weights the same length as cost_list."
            )

        # Check if all costs depend on the same problem
        for cost in self.cost_list:
            if (
                hasattr(cost, "problem")
                and cost.problem is not self.cost_list[0].problem
            ):
                self._different_problems = True

        if not self._different_problems:
            super(WeightedCost, self).__init__(self.cost_list[0].problem)
            self._fixed_problem = self.cost_list[0]._fixed_problem
        else:
            super(WeightedCost, self).__init__()
            self._fixed_problem = False
            for cost in self.cost_list:
                self.parameters.join(cost.parameters)

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
        e = np.empty_like(self.cost_list)

        if not self._fixed_problem and self._different_problems:
            self.parameters.update(values=list(inputs.values()))
        elif not self._fixed_problem:
            self._current_prediction = self.problem.evaluate(inputs)

        for i, cost in enumerate(self.cost_list):
            if not self._fixed_problem and self._different_problems:
                inputs = cost.parameters.as_dict()
                cost._current_prediction = cost.problem.evaluate(inputs)
            else:
                cost._current_prediction = self._current_prediction
            e[i] = cost._evaluate(inputs, grad)

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
        e = np.empty_like(self.cost_list)
        de = np.empty((len(self.parameters), len(self.cost_list)))

        if not self._fixed_problem and self._different_problems:
            self.parameters.update(values=list(inputs.values()))
        elif not self._fixed_problem:
            self._current_prediction, self._current_sensitivities = (
                self.problem.evaluateS1(inputs)
            )

        for i, cost in enumerate(self.cost_list):
            if not self._fixed_problem and self._different_problems:
                inputs = cost.parameters.as_dict()
                cost._current_prediction, cost._current_sensitivities = (
                    cost.problem.evaluateS1(inputs)
                )
            else:
                cost._current_prediction, cost._current_sensitivities = (
                    self._current_prediction,
                    self._current_sensitivities,
                )
            e[i], de[:, i] = cost._evaluateS1(inputs)

        e = np.dot(e, self.weights)
        de = np.dot(de, self.weights)

        return e, de
