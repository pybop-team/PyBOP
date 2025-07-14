from copy import copy, deepcopy

import numpy as np
import pybamm

from pybop import Inputs, Parameters


class PybammPipeline:
    """
    A class to build a PyBaMM pipeline for a given model and data, and run the resultant simulation.

    There are two contexts in which this class can be used:
    1. build_on_eval=True: A pybamm model needs to be built multiple times with different parameter
        values, for the case where any parameters is a geometric parameter, which changes the mesh.
    2. build_on_eval=False: A pybamm model needs to be built once, and then run multiple times with
        different input parameters.
    """

    def __init__(
        self,
        model: pybamm.BaseModel,
        geometry: pybamm.Geometry | None = None,
        parameter_values: pybamm.ParameterValues | None = None,
        submesh_types: dict | None = None,
        var_pts: dict | None = None,
        spatial_methods: dict | None = None,
        solver: pybamm.BaseSolver | None = None,
        pybop_parameters: Parameters | None = None,
        t_start: np.number = 0.0,
        t_end: np.number = 1.0,
        t_interp: np.ndarray | None = None,
        initial_state: float | str | None = None,
        build_on_eval: bool = False,
    ):
        """
        Parameters
        ---------
        model : pybamm.BaseModel
            The PyBaMM model to be used.
        geometry: pybamm.Geometry (optional)
            The geometry upon which to solve the model.
        parameter_values : pybamm.ParameterValues (optional)
            Parameters and their corresponding numerical values.
        submesh_types : dict (optional)
            A dictionary of the types of submesh to use on each subdomain.
        var_pts : dict (optional)
            A dictionary of the number of points used by each spatial variable.
        spatial_methods : dict (optional)
            A dictionary of the types of spatial method to use on each.
            domain (e.g. pybamm.FiniteVolume)
        solver : pybamm.BaseSolver (optional)
            The solver to use to solve the model.
        pybop_parameters : pybop.Parameters (optional)
            The parameters to be optimised.
        t_start : number (optional)
            The start time of the simulation.
        t_end : number (optional)
            The end time of the simulation.
        t_interp : np.ndarray (optional)
            The time points at which to interpolate the solution. If None, no interpolation will be done.
        initial_state: float | str (optional)
            The initial state of charge or voltage for the battery model. If float, it will be
            represented as SoC and must be in range 0 to 1. If str, it will be represented as voltage and
            needs to be in the format: "3.4 V".
        build_on_eval : bool
            Boolean to determine if the model will be rebuilt every evaluation. If `initial_state` is provided,
            the model will be rebuilt every evaluation unless `build_on_eval` is `False`, in which case the model
            is built with the parameter values from construction only.
        """
        self._model = model
        self._geometry = geometry or model.default_geometry
        self._parameter_values = parameter_values or model.default_parameter_values
        self._submesh_types = submesh_types or model.default_submesh_types
        self._var_pts = var_pts or model.default_var_pts
        self._spatial_methods = spatial_methods or model.default_spatial_methods
        self._solver = solver or model.default_solver

        self._pybop_parameters = pybop_parameters or Parameters([])
        self._t_start = np.float64(t_start)
        self._t_end = np.float64(t_end)
        self._t_interp = t_interp
        self._initial_state = initial_state
        self._built_model = self._model
        self.requires_rebuild = build_on_eval or self._determine_rebuild()

    def _determine_rebuild(self) -> bool:
        """
        The initial build process, useful to determine if rebuilding will be required.
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
        Checks whether the parameter values required a rebuild. This is reimplemented with only the
        required functionality.
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

            # Also check initial state calculation (shouldn't be needed in future)
            self._set_initial_state()
        except ValueError:
            return True
        return False

    def _process_and_check(self, sym):
        """
        Process and check the geometry for each parameter. This is reimplemented as geometric parameters
        are not currently supported as InputParameters within PyBaMM.
        Credit: PyBaMM Team
        """
        new_sym = self._parameter_values.process_symbol(sym)
        leaves = new_sym.post_order(filter=lambda node: len(node.children) == 0)
        for leaf in leaves:
            if not isinstance(leaf, pybamm.Scalar):
                raise ValueError("Geometry parameters must be Scalars")
        return new_sym

    def rebuild(self, inputs: Inputs) -> None:
        """
        Build the PyBaMM pipeline using the given parameter_values.
        """
        # if there are no parameters to build, just return
        if not self.requires_rebuild:
            return

        # we need to rebuild, so make sure we've got the right number of parameters
        # and set them in the parameters object
        if len(inputs) != len(self._pybop_parameters):
            raise ValueError(
                f"Expected {len(self._pybop_parameters)} parameters, but got {len(inputs)}."
            )

        self._parameter_values.update(inputs)
        self.build()

    def build(self) -> None:
        """
        Build the PyBaMM pipeline using the given parameter_values.
        """
        model = self._model.new_copy()
        geometry = copy(self._geometry)

        # set parameters in place
        self._set_initial_state()
        self._parameter_values.process_model(model)
        self._parameter_values.process_geometry(geometry)

        mesh = pybamm.Mesh(geometry, self._submesh_types, self._var_pts)
        disc = pybamm.Discretisation(mesh, self._spatial_methods, check_model=True)
        disc.process_model(model)
        self._built_model = model

        # reset the solver since we've built a new model
        self._solver = self._solver.copy()

    def solve(self, calculate_sensitivities: bool = False) -> pybamm.Solution:
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
        return self._solver.solve(
            model=self._built_model,
            inputs=self._pybop_parameters.to_dict(),
            t_eval=[self._t_start, self._t_end],
            t_interp=self._t_interp,
            calculate_sensitivities=calculate_sensitivities,
        )

    def _set_initial_state(self) -> None:
        """
        Sets the parameter values which define the initial state of the model.
        """
        if self._initial_state is not None:
            param = self.model.param
            options = self.model.options
            inputs = self._pybop_parameters.to_dict()
            if options["open-circuit potential"] == "MSMR":
                self._parameter_values.set_initial_ocps(
                    self._initial_state, param=param, options=options, inputs=inputs
                )
            elif options["working electrode"] == "positive":
                self._parameter_values.set_initial_stoichiometry_half_cell(
                    self._initial_state, param=param, options=options, inputs=inputs
                )
            else:
                self._parameter_values.set_initial_stoichiometries(
                    self._initial_state, param=param, options=options, inputs=inputs
                )

    @property
    def built_model(self):
        """The built PyBaMM model."""
        return self._built_model

    @property
    def parameter_names(self):
        return self.pybop_parameters.keys()

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
