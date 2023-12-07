import pybamm
from ..base_model import BaseModel


class Thevenin(BaseModel):
    """
    Composition of the PyBaMM Single Particle Model class.

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
