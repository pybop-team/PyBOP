from copy import copy, deepcopy

import numpy as np
import pybamm

from pybop import Parameters


class PybammPipeline:
    """
    A class to build a PyBaMM pipeline for a given model and data, and run the resultant simulation.

    There are two contexts in which this class can be used:
    1. A pybamm model needs to be built once, and then run multiple times with different input parameters
    2. A pybamm model needs to be built multiple times with different parameter values,
        for the case where some of the parameters are geometric parameters which change the mesh

    The logic for (1) and (2) occurs within the composed PybammPipeline and happens automatically.
    To override this logic, the argument `build_on_eval` can be set to `True` which will force (2) to
    occur.

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
        initial_state: dict | None = None,
        initial_state_parameters: list | None = None,
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
        initial_state: dict (optional)
            A valid initial state, e.g. the initial state of charge or open-circuit voltage.
        initial_state_parameters: list (optional)
            The names of the parameters which define initial state of the model. Only required for a
            custom model with initial state setting.
        build_on_eval : bool
            Boolean to determine if the model will be rebuilt every evaluation. By default, the model will
            only be rebuilt if needed, for example for an initial state or geometric parameters.
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
        self.initial_state = self._convert_to_pybamm_initial_state(initial_state)
        self._initial_state_parameters = (
            initial_state_parameters or self._get_initial_state_parameters()
        )
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
        for parameter in self._initial_state_parameters:
            parameter_values.update({parameter: "[input]"})

        parameter_values.process_geometry(geometry)
        parameter_values.process_model(model)
        requires_rebuild = self._parameters_require_rebuild(geometry, parameter_values)
        if not requires_rebuild:
            # We can use PyBaMM's InputParameter functionality
            self._parameter_values = parameter_values
        return requires_rebuild

    def _parameters_require_rebuild(self, geometry, parameter_values) -> bool:
        """
        Checks whether the parameter values require a rebuild. This is reimplemented with only the
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

            # Also check initial state calculation
            self._get_initial_state_inputs(parameter_values=parameter_values)
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

    def rebuild(self) -> None:
        """
        Update the parameter values and rebuild the PyBaMM pipeline, if required.
        """
        if not self.requires_rebuild:
            # Parameter values will be passed to the solver as inputs
            return

        # Update the parameter values and build again
        all_inputs = self.get_all_inputs()
        self._parameter_values.update(all_inputs)
        self.build()

    def build(self) -> None:
        """
        Build the PyBaMM pipeline using the given parameter_values.
        """
        model = self._model.new_copy()
        geometry = copy(self._geometry)

        # set parameters in place
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
        all_inputs = self.get_all_inputs() if not self.requires_rebuild else None
        return self._solver.solve(
            model=self._built_model,
            inputs=all_inputs,
            t_eval=[self._t_start, self._t_end],
            t_interp=self._t_interp,
            calculate_sensitivities=calculate_sensitivities,
        )

    def _convert_to_pybamm_initial_state(self, initial_state: dict):
        """
        Convert an initial state of charge into a float and an initial open-circuit
        voltage into a string ending in "V".

        Parameters
        ----------
        initial_state : dict
            A valid initial state, e.g. the initial state of charge or open-circuit voltage.

        Returns
        -------
        float or str
            If float, this value is used as the initial state of charge (as a decimal between 0
            and 1). If str ending in "V", this value is used as the initial open-circuit voltage.

        Raises
        ------
        ValueError
            If the input is not a dictionary with a single, valid key.
        """
        if initial_state is None:
            return None
        elif len(initial_state) > 1:
            raise ValueError("Expecting only one initial state.")
        elif "Initial SoC" in initial_state.keys():
            return initial_state["Initial SoC"]
        elif "Initial open-circuit voltage [V]" in initial_state.keys():
            return str(initial_state["Initial open-circuit voltage [V]"]) + "V"
        else:
            raise ValueError(f'Unrecognised initial state: "{list(initial_state)[0]}"')

    def _get_initial_state_parameters(self) -> list:
        """
        Returns the names of the parameters which define the initial state of the model.
        """
        if self.initial_state is None:
            return []

        ocp_type = self.model.options.get("open-circuit potential", None)
        if ocp_type is None:
            return ["Initial SoC"]  # for equivalent circuit models
        elif ocp_type == "MSMR":
            return [
                "Initial voltage in negative electrode [V]",
                "Initial voltage in positive electrode [V]",
            ]
        elif ocp_type == "positive":
            return [
                "Initial concentration in positive electrode [mol.m-3]",
            ]
        else:
            return [
                "Initial concentration in negative electrode [mol.m-3]",
                "Initial concentration in positive electrode [mol.m-3]",
            ]

    def _get_initial_state_inputs(self, parameter_values=None, inputs=None) -> dict:
        """
        Returns a dictionary of the parameters which define the initial state of the model,
        without modifying the parameter_values attribute.
        """
        if self.initial_state is None:
            return {}

        param = self.model.param
        options = self.model.options
        parameter_values = parameter_values or self._parameter_values.copy()

        # Update parameter values as well as passing inputs to cope with non-/rebuild cases
        parameter_values.update(inputs or {})

        values = parameter_values.set_initial_state(
            self.initial_state,
            param=param,
            inplace=False,
            options=options,
            inputs=inputs,
        )

        return {param: values[param] for param in self._initial_state_parameters}

    def get_all_inputs(self) -> dict:
        """
        Returns a dictionary of all inputs including the parameters which define the
        initial state of the model, without modifying the parameter_values attribute.
        """
        inputs = self._pybop_parameters.to_dict()
        return {**inputs, **self._get_initial_state_inputs(inputs=inputs)}

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
