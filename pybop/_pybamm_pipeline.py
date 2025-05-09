from copy import copy, deepcopy

import numpy as np
import pybamm

from pybop import Inputs, Parameters


class PybammPipeline:
    """
    A class to build a PyBaMM pipeline for a given model and data, and run the resultant simulation.

    There are two contexts in which this class can be used:
    1. A pybamm model needs to be built once, and then run multiple times with different input parameters
    2. A pybamm model needs to be built multiple times with different parameter values,
        for the case where some of the parameters are geometric parameters which change the mesh

    To enable 2., you can pass a list of parameter names to the constructor, these parameters will be set
    before the model is built each time (using the `build` method).
    To enable 1, you can just pass an empty list. The model will be built once and subsequent calls
    to the `build` method will not change the model.
    """

    def __init__(
        self,
        model: pybamm.BaseModel,
        parameter_values: pybamm.ParameterValues = None,
        pybop_parameters: Parameters = None,
        solver: pybamm.BaseSolver = None,
        t_start: np.number = 0,
        t_end: np.number = 1,
        t_interp: np.ndarray = None,
    ):
        """
        Arguments
        ---------
        model : pybamm.BaseModel
            The PyBaMM model to be used.
        parameter_values : pybamm.ParameterValues
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
        self.requires_rebuild = False
        self._model = model
        self._parameter_values = parameter_values
        self._pybop_parameters = pybop_parameters
        self._parameter_names = pybop_parameters.keys()
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
        self.requires_rebuild = self._determine_rebuild()

    def _determine_rebuild(self) -> bool:
        """
        The initial build process, useful to determine if
        rebuilding will be required.
        """
        model = self._model.new_copy()
        parameter_values = self._parameter_values.copy()
        geometry = deepcopy(self._geometry)

        # Apply "[input]"
        for parameter in self._pybop_parameters:
            parameter_values.update({parameter.name: "[input]"})

        parameter_values.process_geometry(geometry)
        parameter_values.process_model(model)
        requires_rebuild = self._parameters_require_rebuild(geometry)
        if not requires_rebuild:
            self._parameter_values = parameter_values
        return requires_rebuild

    def _parameters_require_rebuild(self, geometry) -> bool:
        """
        Checks whether the parameter values required a rebuild.
        This is reimplemented with only the required functionality.
        """
        try:
            # Credit: PyBaMM Team
            for domain in geometry:
                for spatial_variable, spatial_limits in geometry[domain].items():
                    # process tab information if using 1 or 2D current collectors
                    if spatial_variable == "tabs":
                        for _, position_info in spatial_limits.items():
                            for _, sym in position_info.items():
                                self._process_and_check(sym)
                    else:
                        for _, sym in spatial_limits.items():
                            self._process_and_check(sym)
        except ValueError:
            return True
        return False

    def _process_and_check(self, sym):
        """
        Process and check the geometry for each parameter.
        This is reimplemented as geometric parameters are not
        currently supported as InputParameters within Pybop.
        Credit: PyBaMM Team
        """
        new_sym = self._parameter_values.process_symbol(sym)
        leaves = new_sym.post_order(filter=lambda node: len(node.children) == 0)
        for leaf in leaves:
            if not isinstance(leaf, pybamm.Scalar):
                raise ValueError("Geometry parameters must be Scalars")
        return new_sym

    def rebuild(self, params: Inputs) -> None:
        """
        Build the PyBaMM pipeline using the given parameter_values.
        """
        # if there are no parameters to build, just return
        if not self.requires_rebuild:
            return

        # we need to rebuild, so make sure we've got the right number of parameters
        # and set them in the parameters object
        if len(params) != len(self._parameter_names):
            raise ValueError(
                f"Expected {len(self._parameter_names)} parameters, but got {len(params)}."
            )

        for key, value in params.items():
            self._parameter_values[key] = value

        self.build()

    def build(self) -> None:
        """
        Build the PyBaMM pipeline using the given parameter_values.
        """

        model = self._model.new_copy()
        geometry = copy(self._geometry)  # copy?
        self._parameter_values.process_geometry(geometry)
        self._parameter_values.process_model(model)

        mesh = pybamm.Mesh(geometry, self._submesh_types, self._var_pts)
        disc = pybamm.Discretisation(mesh, self._methods, check_model=True)
        disc.process_model(model)
        self._built_model = model

        # reset the solver since we've built a new model
        self._solver = self._solver.copy()

        # self._solver.set_up(model) #Is this required? If so, we need to pass an `inputs` dict

        # TODO: unfortunately, the solver will still call set_up on the model
        # if this is not done, need to fix this in PyBaMM!
        # self._solver._model_set_up.update(  # Noqa: SLF001
        #     {model: {"initial conditions": model.concatenated_initial_conditions}}
        # )

        # self.n_states = self._built_model.len_rhs_and_alg  # len_rhs + len_alg

    @property
    def built_model(self):
        """
        The built PyBaMM model.
        """
        return self._built_model

    @property
    def parameter_names(self):
        return self._parameter_names

    @parameter_names.setter
    def parameter_names(self, parameter_names):
        self._parameter_names = parameter_names

    def solve(self, calculate_sensitivities: bool = False) -> pybamm.Solution:
        """
        Run the simulation using the built model and solver.

        Arguments
        ---------
        calculate_sensitivities : bool
            Whether to calculate sensitivities or not.
        inputs : Inputs

        """
        return self._solver.solve(
            model=self._built_model,
            inputs=self._pybop_parameters.as_dict(),
            t_eval=[self._t_start, self._t_end],
            t_interp=self._t_interp,
            calculate_sensitivities=calculate_sensitivities,
        )
