from typing import Union

import numpy as np

from pybop import BaseCost, Inputs


class CostInterface:
    """
    A base class for the optimisers and samplers that provides a common interface between
    the optimiser/sampler and the cost evaluation.
    """

    def __init__(self):
        pass

    def run(self):
        raise NotImplementedError

    def call_cost(
        self,
        x: Union[Inputs, list],
        cost: Union[BaseCost, callable],
        calculate_grad: bool = False,
        apply_transform: bool = True,
        for_optimiser: bool = True,
    ) -> Union[float, tuple[float, np.ndarray]]:
        """
        Provides the interface between the cost and the optimiser.

        Parameters
        ----------
        x : Inputs or list-like
            The input parameters for which the cost and optionally the gradient
            will be computed.
        cost : pybop.BaseCost or callable
            The objective to be optimised, which can be either a pybop.Cost or callable function.
        calculate_grad : bool, optional, default=False
            If True, both the cost and gradient will be computed. Otherwise, only the
            cost is computed.

        Returns
        -------
        float or tuple
            - If `calculate_grad` is False, returns the computed cost (float).
            - If `calculate_grad` is True, returns a tuple containing the cost (float)
              and the gradient (np.ndarray).
        """
        return cost(
            x,
            calculate_grad=calculate_grad,
            apply_transform=apply_transform,
            for_optimiser=for_optimiser,
        )
