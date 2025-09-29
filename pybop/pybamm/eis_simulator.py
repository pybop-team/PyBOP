import warnings
from copy import copy
from typing import TYPE_CHECKING

import casadi
import numpy as np
import pybamm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

if TYPE_CHECKING:
    from pybop.parameters.parameter import Inputs
from pybop._utils import SymbolReplacer
from pybop.pybamm.simulator import Simulator


class EISSimulator:
    """
    A class to extend a PyBaMM model for EIS, automatically build/rebuild a pybamm.Simulation to obtain
    a built model which can be solved to compute the complex impedance for a given set of frequencies.

    There are two contexts in which this class can be used:
    1. A pybamm model can be built once and then run multiple times with different inputs.
    2. A pybamm model needs to be built and then run for each set of inputs, for example in the case
        where one of the inputs is a geometric parameter which requires a new mesh.

    The logic for (1) and (2) occurs within the composed Simulator and happens automatically.
    To override this logic, the argument `build_every_time` can be set to `True` which will force (2) to
    occur.

    Parameters
    ----------
    model : pybamm.BaseModel
        The PyBaMM model to be used.
    f_eval : list
        The frequencies at which to evaluate the impedance.
    parameter_values : pybamm.ParameterValues, optional
        The parameter values to be used in the model.
    input_parameter_names : list[str], optional
        A list of the input parameter names.
    initial_state : dict, optional
        A valid initial state, e.g. `"Initial open-circuit voltage [V]"` or ``"Initial SoC"`.
        Defaults to None, indicating that the existing initial state of charge (for an ECM)
        or initial concentrations (for an EChem model) will be used.
    solver : pybamm.BaseSolver, optional
        The solver to simulate the composed Simulator. If None, uses `pybop.RecommendedSolver`.
    geometry : pybamm.Geometry, optional
        The geometry upon which to solve the model.
    submesh_types : dict, optional
        A dictionary of the types of submesh to use on each subdomain.
    var_pts : dict, optional
        A dictionary of the number of points used by each spatial variable.
    spatial_methods : dict, optional
        A dictionary of the types of spatial method to use on each domain (e.g. pybamm.FiniteVolume).
    discretisation_kwargs : dict, optional
        Any keyword arguments to pass to the Discretisation class.
        See :class:`pybamm.Discretisation` for details.
    build_every_time : bool, optional
        If True, the model will be rebuilt every evaluation. Otherwise, the need to rebuild will be
        determined automatically.
    """

    def __init__(
        self,
        model: pybamm.BaseModel,
        f_eval: np.ndarray | list[float],
        parameter_values: pybamm.ParameterValues | None = None,
        input_parameter_names: list[str] | None = None,
        initial_state: float | str | None = None,
        solver: pybamm.BaseSolver | None = None,
        geometry: pybamm.Geometry | None = None,
        submesh_types: dict | None = None,
        var_pts: dict | None = None,
        spatial_methods: dict | None = None,
        discretisation_kwargs: dict | None = None,
        build_every_time: bool = False,
    ):
        # Set-up model for EIS
        self._f_eval = f_eval
        model = self.set_up_for_eis(model)
        parameter_values = parameter_values or model.default_parameter_values
        parameter_values["Current function [A]"] = 0

        # Set up a simulation
        self._simulation = Simulator(
            model,
            parameter_values=parameter_values,
            input_parameter_names=input_parameter_names,
            initial_state=initial_state,
            solver=solver,
            geometry=geometry,
            submesh_types=submesh_types,
            var_pts=var_pts,
            spatial_methods=spatial_methods,
            discretisation_kwargs=discretisation_kwargs,
            build_every_time=build_every_time,
        )

        self.debug_mode = False

        # Initialise
        self.M = None
        self._jac = None
        self.b = None

        v_scale = getattr(model.variables["Voltage [V]"], "scale", 1)
        i_scale = getattr(model.variables["Current [A]"], "scale", 1)
        self.z_scale = self._simulation.parameter_values.evaluate(v_scale / i_scale)

    def set_up_for_eis(self, model: pybamm.BaseModel) -> pybamm.BaseModel:
        """
        Set up the model for electrochemical impedance spectroscopy (EIS) simulations.
        This method adds the necessary algebraic equations and variables to the model.
        Originally developed by pybamm-eis: https://github.com/pybamm-team/pybamm-eis

        Parameters
        ----------
        model : pybamm.BaseModel
            The PyBaMM model to be used for EIS simulations.

        Returns
        -------
        pybamm.BaseModel
            The modified model ready for EIS simulations.

        Raises
        ------
        ValueError
            If the model is missing required variables.
        """
        # Verify model has required variables
        required_vars = ["Voltage [V]", "Current [A]"]
        for var in required_vars:
            if var not in model.variables:
                raise ValueError(
                    f"Model must contain variable '{var}' for EIS simulation"
                )

        V_cell = pybamm.Variable("Voltage variable [V]")
        model.variables["Voltage variable [V]"] = V_cell
        V = model.variables["Voltage [V]"]

        # Add algebraic equation for the voltage
        model.algebraic[V_cell] = V_cell - V
        model.initial_conditions[V_cell] = model.param.ocv_init

        # Create the FunctionControl submodel and extract variables
        external_circuit_variables = pybamm.external_circuit.FunctionControl(
            model.param,
            external_circuit_function=None,
            options=model.options,
            control="algebraic",
        ).get_fundamental_variables()

        # Define the variables to replace
        symbol_replacement_map = {}
        for name, variable in external_circuit_variables.items():
            if name in model.variables.keys():
                symbol_replacement_map[model.variables[name]] = variable

        # Don't replace initial conditions, as these should not contain
        # variable objects
        replacer = SymbolReplacer(
            symbol_replacement_map, process_initial_conditions=False
        )
        replacer.process_model(model, inplace=True)

        # Add an algebraic equation for the current density variable
        # External circuit submodels are always equations on the current
        I_cell = model.variables["Current variable [A]"]
        I = model.variables["Current [A]"]
        I_applied = pybamm.FunctionParameter(
            "Current function [A]", {"Time [s]": pybamm.t}
        )
        model.algebraic[I_cell] = I - I_applied
        model.initial_conditions[I_cell] = 0

        return model

    def _model_rebuild(self, inputs: "Inputs") -> None:
        """Update the parameter values and rebuild the EIS model."""
        if self._simulation.requires_model_rebuild:
            self._simulation.rebuild_model(inputs=inputs)
        self._initialise_eis_matrices(inputs=inputs)

    def _initialise_eis_matrices(self, inputs: "Inputs") -> None:
        """
        Initialise the electrochemical impedance spectroscopy (EIS) simulation.
        This method sets up the mass matrix and solver, converts inputs to the appropriate format,
        extracts the necessary attributes from the model, and prepares matrices for the simulation.

        Raises
        ------
        RuntimeError
            If the model hasn't been built yet.
        """
        built_model = self._simulation.built_model
        M = self._simulation.built_model.mass_matrix.entries
        self._simulation.solver.set_up(built_model, inputs=inputs)

        # Convert inputs to casadi format if needed
        casadi_inputs = (
            casadi.vertcat(*inputs.values())
            if inputs is not None and built_model.convert_to_format == "casadi"
            else inputs or []
        )

        # Extract the necessary attributes from the model
        y0 = built_model.concatenated_initial_conditions.evaluate(0, inputs=inputs)
        jac = built_model.jac_rhs_algebraic_eval(0, y0, casadi_inputs).sparse()

        # Convert to Compressed Sparse Column format
        self.M = csc_matrix(M)
        self._jac = csc_matrix(jac)

        # Add forcing to the RHS on the current density
        self.b = np.zeros(y0.shape)
        self.b[-1] = -1

    def solve(
        self, inputs: "Inputs | None" = None, calculate_sensitivities: bool = False
    ) -> np.ndarray:
        """
        Run the EIS simulation to calculate impedance at all specified frequencies.

        Parameters
        ---------
        calculate_sensitivities : bool, optional
            Whether to calculate sensitivities or not (default: False).
            Currently not implemented for EIS.

        Returns
        -------
        np.ndarray
            Complex array containing the impedance values with corresponding frequencies.
        """
        if calculate_sensitivities:
            warnings.warn(
                "Sensitivity calculation not implemented for EIS simulations",
                stacklevel=2,
            )

        return self._catch_errors(inputs)

    def _catch_errors(self, inputs: "Inputs"):
        if not self.debug_mode:
            try:
                return self._solve(inputs)
            except (ZeroDivisionError, RuntimeError, ValueError) as e:
                if isinstance(e, ValueError) and str(e) not in self.exception:
                    raise  # Raise the error if it doesn't match the expected list
                return {"Impedance": np.asarray(np.inf)}

        return self._solve(inputs)

    def _solve(self, inputs: "Inputs"):
        # Always run initialise_eis_matrices, after rebuilding the model if necessary
        self._model_rebuild(inputs)

        zs = [self.calculate_impedance(frequency) for frequency in self._f_eval]

        return {"Impedance": np.asarray(zs) * self.z_scale}

    def calculate_impedance(self, frequency):
        """
        Calculate the impedance for a given frequency.

        This method computes the system matrix, solves the linear system, and calculates
        the impedance based on the solution.

        Parameters
        ----------
        frequency : float
            The frequency at which to calculate the impedance in Hz.

        Returns
        -------
        complex
            The calculated impedance.
        """

        # Compute the system matrix
        A = 1.0j * 2 * np.pi * frequency * self.M - self._jac

        # Solve the system
        x = spsolve(A, self.b)

        # Calculate the impedance (voltage / current)
        return -x[-2] / x[-1]

    @property
    def simulation(self):
        return self._simulation

    @property
    def parameter_values(self):
        return self._simulation.parameter_values

    @property
    def input_parameter_names(self):
        return self._simulation.input_parameter_names

    @property
    def has_sensitivities(self):
        return False

    def copy(self):
        """Return a copy of the simulation."""
        return copy(self)
