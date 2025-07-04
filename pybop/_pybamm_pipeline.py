import multiprocessing as mp
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
        parameter_values: pybamm.ParameterValues | None = None,
        pybop_parameters: Parameters | None = None,
        solver: pybamm.BaseSolver | None = None,
        t_start: np.number = 0.0,
        t_end: np.number = 1.0,
        t_interp: np.ndarray | None = None,
        var_pts: dict | None = None,
        initial_state: float | str | None = None,
        build_on_eval: bool | None = None,
        cost_names: list = None,
    ):
        """
        Parameters
        ---------
        model : pybamm.BaseModel
            The PyBaMM model to be used.
        parameter_values : pybamm.ParameterValues
            The parameters to be used in the model.
        solver : pybamm.BaseSolver
            The solver to be used. If None, the idaklu solver will be used with multithreading.
        t_start : number
            The start time of the simulation.
        t_end : number
            The end time of the simulation.
        t_interp : np.ndarray
            The time points at which to interpolate the solution. If None, no interpolation will be done.
        initial_state: float | str
            The initial state of charge or voltage for the battery model. If float, it will be represented
            as SoC and must be in range 0 to 1. If str, it will be represented as voltage and needs to be in
            the format: "3.4 V".
        build_on_eval : bool
            Boolean to determine if the model will be rebuilt every evaluation. If `initial_state` is provided,
            the model will be rebuilt every evaluation unless `build_on_eval` is `False`, in which case the model
            is built with the parameter values from construction only.
        """
        self._model = model
        self._model.events = []
        self._parameter_values = parameter_values or model.default_parameter_values
        self._pybop_parameters = pybop_parameters or Parameters([])
        self._parameter_names = self.pybop_parameters.keys()
        self._geometry = model.default_geometry
        self._methods = model.default_spatial_methods
        self._threads = self.get_avaliable_thread_count()
        self._t_start = np.float64(t_start)
        self._t_end = np.float64(t_end)
        self._t_interp = t_interp
        self._initial_state = initial_state
        self._built_initial_soc = None
        self._var_pts = var_pts or model.default_var_pts
        self._submesh_types = model.default_submesh_types
        self._built_model = self._model
        self.requires_rebuild = (
            build_on_eval
            if build_on_eval is not None
            else True
            if initial_state is not None
            else self._determine_rebuild()
        )
        self._solver = (
            pybamm.IDAKLUSolver(options={"num_threads": self._threads})
            if solver is None
            else solver
        )

    @staticmethod
    def get_avaliable_thread_count():
        """
        Returns the number of available threads available for processing
        with a lower limit of 1.
        """
        return max(1, mp.cpu_count())

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
        if len(params) != len(self._pybop_parameters):
            raise ValueError(
                f"Expected {len(self._pybop_parameters)} parameters, but got {len(params)}."
            )

        for key, value in params.items():
            self._parameter_values[key] = value

        self.build()

    def build(self) -> None:
        """
        Build the PyBaMM pipeline using the given parameter_values.
        """
        model = self._model.new_copy()

        if self._initial_state is not None:
            self._set_initial_state(model, self._initial_state)

        geometry = copy(self._geometry)
        self._parameter_values.process_geometry(geometry)
        self._parameter_values.process_model(model)

        mesh = pybamm.Mesh(geometry, self._submesh_types, self._var_pts)
        disc = pybamm.Discretisation(mesh, self._methods, check_model=True)
        disc.process_model(model)
        self._built_model = model

        # reset the solver since we've built a new model
        self._solver = self._solver.copy()

    @property
    def built_model(self):
        """
        The built PyBaMM model.
        """
        return self._built_model

    @property
    def parameter_names(self):
        return self._parameter_names

    def solve(self, calculate_sensitivities: bool = False) -> list[pybamm.Solution]:
        """
        Run the simulation using the built model and solver.

        Parameters
        ---------
        calculate_sensitivities : bool
            Whether to calculate sensitivities or not.

        Returns
        -------
        solution : pybamm.Solution
            The pybamm solution object.
        """
        sol = self._solver.solve(
            model=self._built_model,
            inputs=self._pybop_parameters.to_pybamm_multiprocessing(),
            t_eval=[self._t_start, self._t_end],
            t_interp=self._t_interp,
            calculate_sensitivities=calculate_sensitivities,
        )

        if not isinstance(sol, list):
            return [sol]

        return sol

    def _set_initial_state(self, model, initial_state) -> None:
        """
        Sets the initial state of the model.

        Parameters
        ----------
        model : pybamm.Model

        initial_state : float | str
            Can be either a float between 0 and 1 representing the initial SoC,
            or a string representing the initial voltage i.e. "3.4 V"
        """

        options = model.options
        param = model.param
        if options["open-circuit potential"] == "MSMR":
            self._parameter_values.set_initial_ocps(
                initial_state, param=param, options=options
            )
        elif options["working electrode"] == "positive":
            self._parameter_values.set_initial_stoichiometry_half_cell(
                initial_state,
                param=param,
                options=options,
                inputs=self._pybop_parameters.to_dict(),
            )
        else:
            self._parameter_values.set_initial_stoichiometries(
                initial_state,
                param=param,
                options=options,
                inputs=self._pybop_parameters.to_dict(),
            )

        # Save solved initial SOC in case we need to re-build the model
        self._built_initial_state = initial_state

    def set_parameter_value(self, key, value):
        self._parameter_values[key] = value

    @property
    def model(self):
        return self._model

    @property
    def pybop_parameters(self):
        return self._pybop_parameters

    @property
    def parameter_values(self):
        return self._parameter_values

    @property
    def solver(self):
        return self._solver
