import pybop
import pybamm


class BaseModel:
    """
    Base class for PyBOP models
    """

    def __init__(self, name="Base Model"):
        self.name = name
        self.pybamm_model = None
        self.fit_parameters = None
        self.observations = None

    def build(
        self,
        observations=None,
        fit_parameters=None,
        check_model=True,
        init_soc=None,
    ):
        """
        Build the PyBOP model (if not built already).
        For PyBaMM forward models, this method follwos a 
        similar process to pybamm.Simulation.build().
        """
        self.fit_parameters = fit_parameters
        self.observations = observations

        if init_soc is not None:
            self.set_init_soc(init_soc)

        if self._built_model:
            return

        elif self.pybamm_model.is_discretised:
            self._model_with_set_params = self.pybamm_model
            self._built_model = self.pybamm_model
        else:
            self.set_params()
            self._mesh = pybamm.Mesh(self.geometry, self.submesh_types, self.var_pts)
            self._disc = pybamm.Discretisation(self.mesh, self.spatial_methods)
            self._built_model = self._disc.process_model(
                self._model_with_set_params, inplace=False, check_model=check_model
            )

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

    def set_params(self):
        """
        Set the parameters in the model.
        """
        if self.model_with_set_params:
            return

        if self.fit_parameters is not None:
            # set input parameters in parameter set from fitting parameters
            for i in self.fit_parameters.keys():
                self.parameter_set[i] = "[input]"

        if self.observations is not None and self.fit_parameters is not None:
            self.parameter_set["Current function [A]"] = pybamm.Interpolant(
                self.observations["Time [s]"].data,
                self.observations["Current function [A]"].data,
                pybamm.t,
            )
            # Set t_eval
            self.time_data = self._parameter_set["Current function [A]"].x[0]

        self._model_with_set_params = self._parameter_set.process_model(
            self._unprocessed_model, inplace=False
        )
        self._parameter_set.process_geometry(self.geometry)
        self.pybamm_model = self._model_with_set_params

    def simulate(self, inputs, t_eval):
        """
        Run the forward model and return the result in Numpy array format
        aligning with Pints' ForwardModel simulate method.
        """
        if self._built_model is None:
            raise ValueError("Model must be built before calling simulate")
        else:
            if type(inputs) is not dict:
                inputs_dict = {
                    key: inputs[i] for i, key in enumerate(self.fit_parameters)
                }
            return self.solver.solve(
                self.built_model, inputs=inputs_dict, t_eval=t_eval
            )["Terminal voltage [V]"].data

    def predict(self, inputs=None, t_eval=None, parameter_set=None, experiment=None):
        """
        Create a PyBaMM simulation object, solve it, and return a solution object.
        """
        parameter_set = parameter_set or self.parameter_set
        if inputs is not None:
            parameter_set.update(inputs)
        if self._unprocessed_model is not None:
            if experiment is None:
                return pybamm.Simulation(
                    self._unprocessed_model,
                    parameter_values=parameter_set,
                ).solve(t_eval=t_eval)
            else:
                return pybamm.Simulation(
                    self._unprocessed_model,
                    experiment=experiment,
                    parameter_values=parameter_set,
                ).solve()
        else:
            raise ValueError("This sim method currently only supports PyBaMM models")

    def n_parameters(self):
        """
        Returns the dimension of the parameter space.
        """
        return len(self.fit_parameters)

    def n_outputs(self):
        """
        Returns the number of outputs this model has. The default is 1.
        """
        return 1

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
