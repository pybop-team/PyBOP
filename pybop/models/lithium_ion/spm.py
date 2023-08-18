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
        parameter_set=None,
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

        self.parameter_set = parameter_set or self.pybamm_model.default_parameter_values
        self._unprocessed_parameter_set = self.parameter_set

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

    def build_model(
        self,
        observations,
        fit_parameters,
        check_model=True,
        init_soc=None,
    ):
        """
        Build the model (if not built already).
        """
        if init_soc is not None:
            self.set_init_soc(init_soc)

        if self._built_model:
            return

        elif self.pybamm_model.is_discretised:
            self._model_with_set_params = self.pybamm_model
            self._built_model = self.pybamm_model
        else:
            self.set_params(observations, fit_parameters)
            self._mesh = pybamm.Mesh(self.geometry, self.submesh_types, self.var_pts)
            self._disc = pybamm.Discretisation(self.mesh, self.spatial_methods)
            self._built_model = self._disc.process_model(
                self._model_with_set_params, inplace=False, check_model=check_model
            )
            # Set t_eval
            self.time_data = self._parameter_set["Current function [A]"].x[0]

            # Clear solver
            self._solver._model_set_up = {}

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

        param = self.pybamm_model.param
        self.parameter_set = (
            self._unprocessed_parameter_set.set_initial_stoichiometries(
                init_soc, param=param, inplace=False
            )
        )
        # Save solved initial SOC in case we need to rebuild the model
        self._built_initial_soc = init_soc

    def set_params(self, observations, fit_parameters):
        """
        Set the parameters in the model.
        """
        if self.model_with_set_params:
            return

        try:
            self.parameter_set["Current function [A]"] = pybamm.Interpolant(
                observations["Time [s]"].data,
                observations["Current function [A]"].data,
                pybamm.t,
            )
        except:
            raise ValueError("Current function not supplied")

        # set input parameters in parameter set from fitting parameters
        for i in fit_parameters:
            self.parameter_set[i] = "[input]"

        self._model_with_set_params = self._parameter_set.process_model(
            self._unprocessed_model, inplace=False
        )
        self._parameter_set.process_geometry(self.geometry)
        self.pybamm_model = self._model_with_set_params

    def sim(self):
        """
        Simulate the model
        """

    @property
    def built_model(self):
        return self._built_model

    @property
    def parameter_set(self):
        return self._parameter_set

    @parameter_set.setter
    def parameter_set(self, parameter_set):
        self._parameter_set = parameter_set.copy()

    @property
    def model_with_set_params(self):
        return self._model_with_set_params

    @property
    def built_model(self):
        return self._built_model

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, geometry):
        self._geometry = geometry.copy()

    @property
    def submesh_types(self):
        return self._submesh_types

    @submesh_types.setter
    def submesh_types(self, submesh_types):
        self._submesh_types = submesh_types.copy()

    @property
    def mesh(self):
        return self._mesh

    @property
    def var_pts(self):
        return self._var_pts

    @var_pts.setter
    def var_pts(self, var_pts):
        self._var_pts = var_pts.copy()

    @property
    def spatial_methods(self):
        return self._spatial_methods

    @spatial_methods.setter
    def spatial_methods(self, spatial_methods):
        self._spatial_methods = spatial_methods.copy()

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, solver):
        self._solver = solver.copy()
