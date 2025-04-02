import pybamm
import numpy as np


class PybammPipeline:
    """
    A class to build a PyBaMM pipeline for a given model and data, and run the resultant simulation.

    There are two contexts in which this class can be used:
    1. A pybamm model needs to be built once, and then run multiple times with different input parameters
    2. A pybamm model needs to be built multiple times with different parameter values,
        for the case where some of the parameters are geometric parameters which change the mesh

    To enable 2., you can pass a list of parameter names to the constructor, these parameters will be set
    before the model is built each time (using the `rebuild` method).
    To enable 1, you can just pass an empty list. The model will be built once and subsequent calls
    to the `rebuild` method will not change the model.
    """

    def __init__(
        self,
        model: pybamm.BaseModel,
        parameters: pybamm.ParameterValues = None,
        solver: pybamm.BaseSolver = None,
        t_start: np.number = 0,
        t_end: np.number = 1,
        t_interp: np.ndarray = None,
        rebuild_parameters: list[str] = None,
    ):
        """
        Arguments
        ---------
        model : pybamm.BaseModel
            The PyBaMM model to be used.
        parameters : pybamm.ParameterValues
            The parameters to be used in the model.
        solver : pybamm.BaseSolver
            The solver to be used. If None, the idaklu solver will be used.
        t_start : number
            The start time of the simulation.
        t_end : number
            The end time of the simulation.
        t_interp : np.ndarray
            The time points at which to interpolate the solution. If None, no interpolation will be done.
        rebuild_parameters : list[str]
            The parameters that will be used to rebuild the model. If None, the model will not be rebuilt.
        """
        if rebuild_parameters is None:
            rebuild_parameters = []
        params = np.array([parameters[n] for n in rebuild_parameters], dtype=float)
        self._parameter_names = rebuild_parameters
        self._model = model
        self._parameters = parameters
        self._geometry = model.default_geometry
        self._methods = model.default_spatial_methods
        if solver is None:
            self._solver = pybamm.IDAKLUSolver()
        else:
            self._solver = solver
        self._t_start = t_start
        self._t_end = t_end
        self._t_interp = t_interp
        var = pybamm.standard_spatial_vars
        self._var_pts = {
            var.x_n: 30,
            var.x_s: 30,
            var.x_p: 30,
            var.r_n: 10,
            var.r_p: 10,
        }
        self._submesh_types = model.default_submesh_types
        self._built_model = self._model
        self.rebuild(params)

    def rebuild(self, params: np.ndarray):
        """
        Build the PyBaMM pipeline using the given parameters.
        """
        # if there are no parameters to rebuild, just return
        if len(self._parameter_names) == 0:
            return

        # we need to rebuild, so make sure we've got the right number of parameters
        # and set them in the parameters object
        if len(params) != len(self._parameter_names):
            raise ValueError(
                f"Expected {len(self._parameters)} parameters, but got {len(params)}."
            )
        for name, value in zip(self._parameter_names, params):
            self._parameters[name] = value

        model = self._model.deep_copy()
        geometry = self._geometry.deep_copy()
        self._parameters.process_geometry(geometry)
        self._parameters.process_model(model)

        mesh = pybamm.Mesh(geometry, self._submesh_types, self._var_pts)
        disc = pybamm.Discretisation(mesh, self._methods)
        disc.process_model(model)
        self._built_model = model
        self._solver.set_up(model)
        # TODO: unfortunately, the solver will still call set_up on the model
        # if this is not done, need to fix this in PyBaMM!
        self._solver._model_set_up.update(
            {model: {"initial conditions": model.concatenated_initial_conditions}}
        )

    @property
    def built_model(self):
        """
        The built PyBaMM model.
        """
        return self._built_model

    def solve(self) -> pybamm.Solution:
        """
        Run the simulation using the built model and solver.
        """
        return self._solver.solve(
            t_eval=[self._t_start, self._t_end], t_interp=self._t_interp
        )
