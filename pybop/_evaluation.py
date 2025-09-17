import numpy as np
from pints import Evaluator as PintsEvaluator

from pybop import BaseCost
from pybop._logging import Logger


class BaseEvaluator(PintsEvaluator):
    """
    Evaluates a function (or callable object) for multiple positions.

    Applies transformations, if any.

    Based on and extends Pints' :class:`Evaluator`.

    Parameters
    ----------
    problem : pybop.Problem
        The problem to be optimised.
    minimise : bool
        If True, the cost function is minimised, otherwise maximisation is performed by
        inverting the sign of the cost function and sensitivities.
    with_sensitivities : bool
        If True, the sensitivities are calculated as well as the cost.
    logger : Logger
        The logging object to record the parameter and cost values.
    """

    def __init__(
        self,
        cost: BaseCost,
        minimise: bool,
        with_sensitivities: bool,
        logger: Logger,
    ):
        self.transformation = cost.parameters.transformation
        self.cost = cost

        # Choose which function to evaluate
        if minimise and with_sensitivities:

            def fun(x_search):
                x_model = [self.transformation.to_model(x) for x in x_search]
                if len(x_model) == 0:
                    return np.empty(0), np.empty(0)

                inputs_list = self.cost.parameters.to_inputs(x_model)
                cost, grad = [], []
                for inputs in inputs_list:
                    out = self.cost.single_call(inputs, calculate_grad=True)
                    cost.append(out[0])
                    grad.append(out[1])
                cost, grad = np.asarray(cost), np.asarray(grad)

                # Apply the inverse parameter transformation to the gradient
                for i, x in enumerate(x_search):
                    jac = self.transformation.jacobian(x)
                    grad[i] = np.matmul(grad[i], jac)

                logger.extend_log(x_search=x_search, x_model=x_model, cost=cost)

                if len(cost) == 1:
                    return cost, grad.reshape(-1)
                return cost, grad

        elif minimise:

            def fun(x_search):
                x_model = [self.transformation.to_model(x) for x in x_search]
                if len(x_model) == 0:
                    return np.empty(0)

                inputs_list = self.cost.parameters.to_inputs(x_model)
                cost = []
                for inputs in inputs_list:
                    cost.append(self.cost.single_call(inputs, calculate_grad=False))
                cost = np.atleast_1d(cost)

                logger.extend_log(x_search=x_search, x_model=x_model, cost=cost)
                return cost

        # Otherwise, perform maximisation by minimising the negative of the cost
        elif with_sensitivities:

            def fun(x_search):
                x_model = [self.transformation.to_model(x) for x in x_search]
                if len(x_model) == 0:
                    return np.empty(0), np.empty(0)

                inputs_list = self.cost.parameters.to_inputs(x_model)
                cost, grad = [], []
                for inputs in inputs_list:
                    out = self.cost.single_call(inputs, calculate_grad=True)
                    cost.append(out[0])
                    grad.append(out[1])
                cost, grad = np.asarray(cost), np.asarray(grad)

                # Apply the inverse parameter transformation to the gradient
                for i, x in enumerate(x_search):
                    jac = self.transformation.jacobian(x)
                    grad[i] = np.matmul(grad[i], jac)

                logger.extend_log(x_search=x_search, x_model=x_model, cost=cost)

                if len(cost) == 1:
                    return -cost, -grad.reshape(-1)
                return -cost, -grad

        else:

            def fun(x_search):
                x_model = [self.transformation.to_model(x) for x in x_search]
                if len(x_model) == 0:
                    return np.empty(0), np.empty(0)

                inputs_list = self.cost.parameters.to_inputs(x_model)
                cost = []
                for inputs in inputs_list:
                    cost.append(self.cost.single_call(inputs, calculate_grad=False))
                cost = np.atleast_1d(cost)

                logger.extend_log(x_search=x_search, x_model=x_model, cost=cost)
                return -cost

        # Pass function to PintsEvaluator
        super().__init__(fun)


class PopulationEvaluator(BaseEvaluator):
    """
    Evaluates a function (or callable object) for multiple positions.

    Parameters
    ----------
    function : callable
        The function to evaluate. This function should accept a list-like
        object of positions to be evaluated.
    args : sequence, optional
        A sequence containing extra arguments to be passed to the function.
        If specified, the function will be called as `function(x, *args)`.
    """

    def _evaluate(self, positions):
        return self._function(positions, *self._args)


class SequentialEvaluator(BaseEvaluator):
    """
    Evaluates a function (or callable object) for a list of input values, and
    returns a list containing the calculated function evaluations.

    Parameters
    ----------
    function : callable
        The function to evaluate.
    args : sequence
        An optional tuple containing extra arguments to ``f``. If ``args`` is
        specified, ``f`` will be called as ``f(x, *args)``.
    """

    def _evaluate(self, positions):
        scores = [self._function([x], *self._args) for x in positions]

        # Non-sensitivity costs should be a singular dimension array
        if not isinstance(scores[0], tuple):
            return np.asarray(scores).reshape(-1)

        return scores


class ScalarEvaluator(BaseEvaluator):
    """
    Evaluates a function (or callable object) for a single input.

    Parameters
    ----------
    function : callable
        The function to evaluate. This function should accept an input and
        optionally additional arguments, returning either a single value or a tuple.
    args : sequence, optional
        A sequence containing extra arguments to be passed to the function.
        If specified, the function will be called as `function(x, *args)`.
    """

    def _evaluate(self, x):
        return self._function([x], *self._args)
