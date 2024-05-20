import pybamm

from .base_ecm import ECircuitModel


class Thevenin(ECircuitModel):
    """
    The Thevenin class represents an equivalent circuit model based on the Thevenin model in PyBaMM.

    This class encapsulates the PyBaMM equivalent circuit Thevenin model, providing an interface
    to define the parameters, geometry, submesh types, variable points, spatial methods, and solver
    to be used for simulations.

    Parameters
    ----------
    name : str, optional
        A name for the model instance. Defaults to "Equivalent Circuit Thevenin Model".
    **model_kwargs : optional
        Valid PyBaMM model option keys and their values.
    """

    def __init__(
        self,
        name="Equivalent Circuit Thevenin Model",
        **model_kwargs,
    ):
        super().__init__(
            pybamm_model=pybamm.equivalent_circuit.Thevenin, name=name, **model_kwargs
        )

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
