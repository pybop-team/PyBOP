import pybop
import pybamm
from ..BaseModel import BaseModel


class SPM(BaseModel):
    """
    Composition of the SPM class in PyBaMM.
    """

    def __init__(
        self,
        name="Single Particle Model",
        geometry=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
    ):
        super().__init__()
        self.pybamm_model = pybamm.lithium_ion.SPM()
        self._unprocessed_model = self.pybamm_model
        self.name = name
        self.parameter_set = self.pybamm_model.default_parameter_values
        self._model_with_set_params = None
        self._built_model = None
        self._geometry = geometry or self.pybamm_model.default_geometry
        self._submesh_types = submesh_types or self.pybamm_model.default_submesh_types
        self._var_pts = var_pts or self.pybamm_model.default_var_pts
        self._spatial_methods = (
            spatial_methods or self.pybamm_model.default_spatial_methods
        )
        self.solver = solver or self.pybamm_model.default_solver

    def build_model(
        self,
        check_model=True,
        init_soc=None,
    ):
        """
        Build the model (if not built already).
        """
        if init_soc is not None:
            self.set_init_soc(init_soc)  # define this function

        if self._built_model:
            return

        elif self.pybamm_model.is_discretised:
            self.pybamm_model._model_with_set_params = self.pybamm_model
            self.pybamm_model._built_model = self.pybamm_model
        else:
            self.set_params()
            self._mesh = pybamm.Mesh(self._geometry, self._submesh_types, self._var_pts)
            self._disc = pybamm.Discretisation(self._mesh, self._spatial_methods)
            self._built_model = self._disc.process_model(
                self._model_with_set_params, inplace=False, check_model=check_model
            )

            # Clear solver
            self.solver._model_set_up = {}

    def set_init_soc(self, init_soc):
        """
        Set the initial state of charge.
        """
        if self._built_initial_soc != init_soc:
            # reset
            self._model_with_set_params = None
            self._built_model = None
            self.op_conds_to_built_models = None
            self.op_conds_to_built_solvers = None

        param = self.model.pybamm_model.param
        self.parameter_set = (
            self._unprocessed_parameter_set.set_initial_stoichiometries(
                init_soc, param=param, inplace=False
            )
        )
        # Save solved initial SOC in case we need to re-build the model
        self._built_initial_soc = init_soc

    def set_params(self):
        """
        Set the parameters in the model.
        """
        if self._model_with_set_params:
            return

        self._model_with_set_params = self.parameter_set.process_model(
            self._unprocessed_model, inplace=False
        )
        self.parameter_set.process_geometry(self._geometry)
        self.pybamm_model = self._model_with_set_params

    def sim(self):
        """
        Simulate the model
        """
