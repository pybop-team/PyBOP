from pybamm import equivalent_circuit as pybamm_equivalent_circuit

from pybop.models.empirical.base_ecm import ECircuitModel


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
        Valid PyBaMM model option keys and their values, for example:
        parameter_set : pybamm.ParameterValues or dict, optional
            The parameters for the model. If None, default parameters provided by PyBaMM are used.
        geometry : dict, optional
            The geometry definitions for the model. If None, default geometry from PyBaMM is used.
        submesh_types : dict, optional
            The types of submeshes to use. If None, default submesh types from PyBaMM are used.
        var_pts : dict, optional
            The discretization points for each variable in the model. If None, default points from PyBaMM are used.
        spatial_methods : dict, optional
            The spatial methods used for discretization. If None, default spatial methods from PyBaMM are used.
        solver : pybamm.Solver, optional
            The solver to use for simulating the model. If None, the default solver from PyBaMM is used.
        build : bool, optional
            If True, the model is built upon creation (default: False).
        options : dict, optional
            A dictionary of options to customise the behaviour of the PyBaMM model.
    """

    def __init__(
        self,
        name="Equivalent Circuit Thevenin Model",
        eis=False,
        **model_kwargs,
    ):
        super().__init__(
            pybamm_model=pybamm_equivalent_circuit.Thevenin,
            name=name,
            eis=eis,
            **model_kwargs,
        )
