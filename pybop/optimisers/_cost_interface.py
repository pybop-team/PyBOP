from typing import Union

import numpy as np

from pybop import BaseCost, Inputs


class CostInterface:
    """
    A base class for the optimisers and samplers that provides a common interface between
    the optimiser/sampler and the cost evaluation.
    """

    def __init__(self):
        self._transformation = None

    def run(self):
        raise NotImplementedError

    def apply_transformation(self, values):
        """Apply transformation if it exists."""
        if self._transformation:
            return [self._transformation.to_model(value) for value in values]
        return values

    def call_cost(
        self,
        x: Union[Inputs, list],
        cost: Union[BaseCost, callable],
        calculate_grad: bool = False,
        for_optimiser: bool = True,
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
        # Apply any transformation
        model_x = self.apply_transformation([x])[0]

        if calculate_grad:
            cost, grad = cost.single_call(
                model_x,
                calculate_grad=calculate_grad,
                for_optimiser=for_optimiser,
            )

            # Compute gradient with respect to the search parameters
            if self._transformation is not None:  #  and np.isfinite(cost):
                jac = self.transformation.jacobian(x)
                grad = np.matmul(grad, jac)

            return cost, grad

        return cost(model_x, for_optimiser=for_optimiser)
    
    @property
    def transformation(self):
        return self._transformation
