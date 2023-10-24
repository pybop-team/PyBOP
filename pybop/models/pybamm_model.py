import pybop
import pybamm
from pybop import BaseModel


class PybammModel(BaseModel):
    """
    Wrapper class for PyBaMM model classes. Extends the BaseModel class.

    """

    def __init__(self):
        super().__init__()
        self.pybamm_model = None

    def build(
        self,
        dataset=None,
        experiment=None,
        parameters=None,
        check_model=True,
        init_soc=None,
    ):
        """
        Build the model (if not built already).

        Specifiy either a dataset or an experiment.
        """
        self.dataset = dataset
        self.experiment = experiment
        self.parameters = parameters

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
        Set each parameter in the model either equal to its value
        or mark it as an input.
        """
        if self.model_with_set_params:
            return

        if self.parameters is not None:
            # Set input parameters in parameter set from fitting parameters
            for i, Param in enumerate(self.parameters):
                self.parameter_set[Param.name] = "[input]"

        if self.dataset is not None:
            self.parameter_set["Current function [A]"] = pybamm.Interpolant(
                self.dataset["Time [s]"].data,
                self.dataset["Current function [A]"].data,
                pybamm.t,
            )
            # Set times
            self.times = self._parameter_set["Current function [A]"].x[0]

        self._model_with_set_params = self._parameter_set.process_model(
            self._unprocessed_model, inplace=False
        )
        self._parameter_set.process_geometry(self.geometry)
        self.pybamm_model = self._model_with_set_params

    def simulate(self, parameters=None, times=None, experiment=None):
        """
        Run the forward model and return the result in Numpy array format
        aligning with Pints' ForwardModel simulate method.

        Inputs
        ----------
        parameters : A collection of fitting parameters to pass to the model when
            solving
        times : numeric type, optional
            The times (in seconds) at which to compute the solution. Can be
            provided as an array of times at which to return the solution, or as a
            list `[t0, tf]` where `t0` is the initial time and `tf` is the final time.
            If provided as a list the solution is returned at 100 points within the
            interval `[t0, tf]`.

            If not using an experiment or running a drive cycle simulation (current
            provided as data) `times` *must* be provided.

            If running an experiment the values in `times` are ignored, and the
            solution times are specified by the experiment.

            If None and the parameter "Current function [A]" is read from data
            (i.e. drive cycle simulation) the model will be solved at the times
            provided in the data.
        experiment : of the PyBaMM Experiment class (for PyBaMM models only)
        """

        # Run the simulation
        prediction = self._simulate(parameters, times, experiment)

        return prediction

    def _simulate(self, parameters, times, experiment):
        """
        Create a PyBaMM simulation object and run it.
        """
        if self.pybamm_model is None:
            raise ValueError("This sim method currently only supports PyBaMM models")

        else:
            # Build the model if necessary
            if self._built_model is None:
                self.build()

            # Pass the input parameters
            if parameters is not None:
                inputs = {}
                for i, Param in enumerate(parameters):
                    inputs[Param.name] = Param.value

            # Define the simulation
            if experiment is None:
                sim = pybamm.Simulation(self.pybamm_model)
                self.times = self.dataset["Time [s]"].data
                t_eval = times or self.times
            else:
                sim = pybamm.Simulation(self.pybamm_model, experiment=experiment)
                t_eval = None

            # Run the simulation
            if parameters is None:
                prediction = sim.solve(t_eval=t_eval)
            else:
                prediction = sim.solve(t_eval=t_eval, inputs=inputs)

            return prediction

    def n_parameters(self):
        """
        Returns the dimension of the parameter space.
        """
        return len(self.parameters)

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
