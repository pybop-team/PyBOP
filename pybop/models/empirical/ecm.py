import pybamm
from ..base_model import BaseModel


class Thevenin(BaseModel):
    """
    The Thevenin class represents an equivalent circuit model based on the Thevenin model in PyBaMM.

    This class encapsulates the PyBaMM equivalent circuit Thevenin model, providing an interface
    to define the parameters, geometry, submesh types, variable points, spatial methods, and solver
    to be used for simulations.

    Parameters
    ----------
    name : str, optional
        A name for the model instance. Defaults to "Equivalent Circuit Thevenin Model".
    parameter_set : dict or None, optional
        A dictionary of parameters to be used for the model. If None, the default parameters from PyBaMM are used.
    geometry : dict or None, optional
        The geometry definitions for the model. If None, the default geometry from PyBaMM is used.
    submesh_types : dict or None, optional
        The types of submeshes to use. If None, the default submesh types from PyBaMM are used.
    var_pts : dict or None, optional
        The number of points for each variable in the model to define the discretization. If None, the default is used.
    spatial_methods : dict or None, optional
        The spatial methods to be used for discretization. If None, the default spatial methods from PyBaMM are used.
    solver : pybamm.Solver or None, optional
        The solver to use for simulating the model. If None, the default solver from PyBaMM is used.
    options : dict or None, optional
        A dictionary of options to pass to the PyBaMM Thevenin model.
    **kwargs :
        Additional arguments passed to the PyBaMM Thevenin model constructor.
    """

    def __init__(
        self,
        name="Equivalent Circuit Thevenin Model",
        parameter_set=None,
        geometry=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
        options=None,
        **kwargs,
    ):
        super().__init__()
        self.pybamm_model = pybamm.equivalent_circuit.Thevenin(
            options=options, **kwargs
        )
        self._unprocessed_model = self.pybamm_model
        self.name = name

        if isinstance(parameter_set, dict):
            self.default_parameter_values = pybamm.ParameterValues(parameter_set)
            self._parameter_set = self.default_parameter_values
        else:
            self.default_parameter_values = self.pybamm_model.default_parameter_values
            self._parameter_set = (
                parameter_set or self.pybamm_model.default_parameter_values
            )

        self._unprocessed_parameter_set = self._parameter_set

        self.geometry = geometry or self.pybamm_model.default_geometry
        self.submesh_types = submesh_types or self.pybamm_model.default_submesh_types
        self.var_pts = var_pts or self.pybamm_model.default_var_pts
        self.spatial_methods = (
            spatial_methods or self.pybamm_model.default_spatial_methods
        )
        self.solver = solver or self.pybamm_model.default_solver

        self._model_with_set_params = None
        self._built_model = None
        self._built_initial_soc = None
        self._mesh = None
        self._disc = None
