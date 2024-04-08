from ..base_model import BaseModel


class ECircuitModel(BaseModel):
    """
    Overwrites and extends `BaseModel` class for circuit-based PyBaMM models.
    """

    def __init__(self, name, parameter_set):
        super().__init__(name, parameter_set)

    def _check_params(self, inputs=None, allow_infeasible_solutions=True):
        """
        Check the compatibility of the model parameters.

        Parameters
        ----------
        inputs : dict
            The input parameters for the simulation.
        allow_infeasible_solutions : bool, optional
            If True, infeasible parameter values will be allowed in the optimisation (default: True).

        Returns
        -------
        bool
            A boolean which signifies whether the parameters are compatible.

        """
        return True
