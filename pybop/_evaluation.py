import numpy as np
from pints import Evaluator as PintsEvaluator

from pybop._logging import Logger
from pybop.problems.base_problem import Problem


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
        problem: Problem,
        minimise: bool,
        with_sensitivities: bool,
        logger: Logger,
    ):
        self.transformation = problem.params.transformation

        # Choose which function to evaluate
        if minimise and with_sensitivities:

            def fun(x_search):
                x_model = [self.transformation.to_model(x) for x in x_search]
                problem.set_params(x_model)
                cost, grad = problem.run_with_sensitivities()

                # Apply the inverse parameter transformation to the gradient
                if len(x_model) == 1:
                    jac = self.transformation.jacobian(x_model[0])
                    grad = np.matmul(grad, jac)
                else:
                    for i in range(grad.shape[0]):
                        jac = self.transformation.jacobian(x_model[i])
                        grad[i] = np.matmul(grad[i], jac)

                logger.extend_log(x_search=x_search, x_model=x_model, cost=cost)
                return cost, grad

        elif minimise:

            def fun(x_search):
                x_model = [self.transformation.to_model(x) for x in x_search]
                problem.set_params(x_model)
                cost = problem.run()
                logger.extend_log(x_search=x_search, x_model=x_model, cost=cost)
                return cost

        # Otherwise, perform maximisation by minimising the negative of the cost
        elif with_sensitivities:

            def fun(x_search):
                x_model = [self.transformation.to_model(x) for x in x_search]
                problem.set_params(x_model)
                neg_cost, neg_grad = problem.run_with_sensitivities()
                cost, grad = -neg_cost, -neg_grad

                # Apply the inverse parameter transformation to the gradient
                if len(x_model) == 1:
                    jac = self.transformation.jacobian(x_model[0])
                    grad = np.matmul(grad, jac)
                else:
                    for i in range(grad.shape[0]):
                        jac = self.transformation.jacobian(x_model[i])
                        grad[i] = np.matmul(grad[i], jac)

                logger.extend_log(x_search=x_search, x_model=x_model, cost=cost)
                return cost, grad

        else:

            def fun(x_search):
                x_model = [self.transformation.to_model(x) for x in x_search]
                problem.set_params(x_model)
                cost = -problem.run()
                logger.extend_log(x_search=x_search, x_model=x_model, cost=cost)
                return cost

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
