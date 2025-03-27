from typing import Optional, Union

import numpy as np

from pybop import BaseCost, Inputs, Transformation


class CostInterface:
    """
    A base class for the optimisers and samplers that provides a common interface between
    the optimiser/sampler and the cost evaluation.
    """

    def __init__(
        self, transformation: Optional[Transformation] = None, invert_cost: bool = False
    ):
        self.invert_cost = invert_cost
        self.transformation = transformation

    def transform_values(self, values):
        """Apply transformation if it exists."""
        if self._transformation:
            return self._transformation.to_model(values)
        return values

    def transform_list_of_values(self, list_of_values):
        """Apply transformation if it exists."""
        if self._transformation:
            return [self._transformation.to_model(values) for values in list_of_values]
        return list_of_values

    def _inverts_cost(self, cost):
        """
        Returns the true cost if the optimiser is operating in the inverted space
        else returns the cost as is.
        """
        return [v * (-1 if self.invert_cost else 1) for v in cost]

    def call_cost(
        self,
        x: Union[Inputs, list],
        cost: Union[BaseCost, callable],
        calculate_grad: bool = False,
    ) -> Union[float, tuple[float, np.ndarray]]:
        """
        Provides the interface between the cost and the optimiser.

        Applies any transformation to the input parameter values, calls the cost function, inverts
        the sign of the cost and gradient if the target is maximisation and reverses the effect of
        any transformation on the gradient.

        Parameters
        ----------
        x : Inputs or list-like
            The input parameters for which the cost and optionally the gradient will be computed.
        cost : pybop.BaseCost or callable
            The objective to be optimised, which can be either a pybop.Cost or callable function.
        calculate_grad : bool, optional, default=False
            If True, both cost and gradient will be computed. Otherwise, only the cost is computed.

        Returns
        -------
        float or tuple
            - If `calculate_grad` is False, returns the computed cost (float).
            - If `calculate_grad` is True, returns a tuple containing the cost (float)
              and the gradient (np.ndarray).
        """
        model_x = self.transform_values(x)

        sign = -1 if self.invert_cost else 1

        if calculate_grad:
            cost, grad = cost.single_call(model_x, calculate_grad=calculate_grad)

            # Compute gradient with respect to the search parameters
            if self._transformation is not None:
                jac = self.transformation.jacobian(x)
                grad = np.matmul(grad, jac)

            return cost * sign, grad * sign

        return cost(model_x) * sign

    @property
    def transformation(self):
        return self._transformation

    @transformation.setter
    def transformation(self, transformation: Transformation):
        self._transformation = transformation
