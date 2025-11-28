import warnings
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import casadi
import numpy as np
import pybamm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

if TYPE_CHECKING:
    from pybop.parameters.parameter import Inputs
from pybop._dataset import Dataset
from pybop._utils import FailedSolution, SymbolReplacer
from pybop.parameters.parameter import Parameter, Parameters
from pybop.pybamm.simulator import Simulator
from pybop.simulators.base_simulator import BaseSimulator, Solution


@dataclass
class TimeSeriesState:
    """
    The current state of a time series model that is a PyBaMM model.
    """

    sol: pybamm.Solution
    inputs: "Inputs"
    t: float = 0.0

    def as_ndarray(self) -> np.ndarray:
        ncol = self.sol.y.shape[1]
        if ncol > 1:
            y = self.sol.y[:, -1]
        else:
            y = self.sol.y
        if isinstance(y, casadi.DM):
            y = y.full()
        return y

    def __len__(self):
        return self.sol.y.shape[0]


class EISSimulator(BaseSimulator):
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
    initial_state : dict, optional
        A valid initial state, e.g. `"Initial open-circuit voltage [V]"` or ``"Initial SoC"`.
        Defaults to None, indicating that the existing initial state of charge (for an ECM)
        or initial concentrations (for an EChem model) will be used.
    protocol : Dataset | np.ndarray, optional
        A 1D array of values or dataset containing the time points at which to simulate
        operando EIS. Defaults to None, corresponding to stationary EIS at time t=0, with I=0.
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
        initial_state: float | str | None = None,
        protocol: Dataset | np.ndarray | None = None,
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
        parameter_values = parameter_values or model.default_parameter_values
        if protocol is None:  # perform stationary EIS by default
            parameter_values["Current function [A]"] = 0
            initial_current = 0
        elif isinstance(protocol, pybamm.Experiment):
            raise ValueError("EISSimulator cannot simulate a pybamm.Experiment.")
        elif (
            isinstance(protocol, Dataset)
            and "Current function [A]" in protocol.data.keys()
        ):
            parameter_values["Current function [A]"] = pybamm.Interpolant(
                protocol["Time [s]"], protocol["Current function [A]"], pybamm.t
            )
            initial_current = protocol["Current function [A]"][0]
        model = self.set_up_for_eis(model, initial_current=float(initial_current))

        # Unpack the uncertain parameters from the parameter values
        parameters = Parameters()
        for name, param in parameter_values.items():
            if isinstance(param, Parameter):
                parameters.add(name, param)
        super().__init__(parameters=parameters)

        # Set up a simulation
        self._simulation = Simulator(
            model,
            parameter_values=parameter_values,
            initial_state=initial_state,
            protocol=protocol,
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
        self._mass = None
        self._jac = None
        self.b = None

        v_scale = getattr(model.variables["Voltage [V]"], "scale", 1)
        i_scale = getattr(model.variables["Current [A]"], "scale", 1)
        self.z_scale = self._simulation.parameter_values.evaluate(v_scale / i_scale)

    def set_up_for_eis(
        self, model: pybamm.BaseModel, initial_current: float
    ) -> pybamm.BaseModel:
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
        model.initial_conditions[I_cell] = initial_current

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
        M = built_model.mass_matrix.entries
        self._simulation.solver.set_up(built_model, inputs=inputs)

        # Convert inputs to casadi format if needed
        casadi_inputs = (
            casadi.vertcat(*inputs.values())
            if inputs is not None and built_model.convert_to_format == "casadi"
            else inputs or []
        )

        ## Stationary EIS
        # Extract the necessary attributes from the model
        y = built_model.concatenated_initial_conditions.evaluate(0, inputs=inputs)
        J = built_model.jac_rhs_algebraic_eval(0, y, casadi_inputs).sparse()

        # Convert to Compressed Sparse Column format
        self._mass = csc_matrix(M)
        self._jac = csc_matrix(J)

        # Add forcing to the RHS on the current density
        self.b = np.zeros(y.shape)
        self.b[-1] = -1

        ## Operando EIS
        if self.time_data is not None:
            # Initial state
            t = np.asarray([0])
            inputs = inputs or {}
            sol = pybamm.Solution([t], [y], built_model, inputs)
            state = TimeSeriesState(sol=sol, inputs=inputs, t=t)

            self._jac_at_time_t = [self._jac]
            for t in self.time_data[1:]:
                # Step forwards in time
                dt = (t - state.t).item()
                new_sol = self._simulation.solver.step(
                    state.sol, built_model, dt, inputs=state.inputs, save=False
                )
                state = TimeSeriesState(sol=new_sol, inputs=state.inputs, t=t)

                # Extract necessary attributes from the model
                y = state.as_ndarray()
                J = built_model.jac_rhs_algebraic_eval(t, y, casadi_inputs).sparse()

                if np.abs(y[-1]) > 1e-10:
                    warnings.warn(
                        f"The current is not zero at the requested EIS point at V={y[-2]} V.",
                        stacklevel=2,
                    )

                # Convert to Compressed Sparse Column format
                self._jac_at_time_t.append(csc_matrix(J))

    def solve(
        self,
        inputs: "Inputs | list[Inputs] | None" = None,
        calculate_sensitivities: bool = False,
    ) -> Solution | list[Solution]:
        """
        Run the EIS simulation for one or more sets of inputs and return the result(s).

        Parameters
        ----------
        inputs : Inputs | list[Inputs], optional
            Input parameters (default: None).
        calculate_sensitivities : bool
            Whether to also return the sensitivities (default: False).
            Currently not implemented for EIS.

        Returns
        -------
        Solution | list[Solution]
            Complex impedance results.
        """
        if calculate_sensitivities:
            warnings.warn(
                "Sensitivity calculation not implemented for EIS simulations",
                stacklevel=2,
            )

        if not isinstance(inputs, list):
            return self._catch_errors([inputs])[0]

        return self._catch_errors(inputs)

    def batch_solve(
        self, inputs: "list[Inputs]" = None, calculate_sensitivities: bool = False
    ) -> list[Solution | FailedSolution]:
        """
        Run the EIS simulation for each set of inputs and return dict-like results.

        Parameters
        ----------
        inputs : list[Inputs]
            A list of input parameters.
        calculate_sensitivities : bool
            Whether to calculate sensitivities (default: False).
            Currently not implemented for EIS.

        Returns
        -------
        list[Solution]
            A list of len(inputs) containing the complex impedance results.
        """
        if calculate_sensitivities:
            warnings.warn(
                "Sensitivity calculation not implemented for EIS simulations",
                stacklevel=2,
            )

        return self._catch_errors(inputs)

    def _catch_errors(self, inputs: "list[Inputs]") -> list[Solution | FailedSolution]:
        if not self.debug_mode:
            simulations = []
            for x in inputs:
                try:
                    simulations.append(self._solve(x))
                except (ZeroDivisionError, RuntimeError, ValueError) as e:
                    if (
                        isinstance(e, ValueError)
                        and str(e) not in self._simulation.exception
                    ):
                        raise  # Raise the error if it doesn't match the expected list
                    simulations.append(
                        FailedSolution(["Impedance"], [k for k in x.keys()])
                    )
            return simulations

        simulations = []
        for x in inputs:
            simulations.append(self._solve(x))
        return simulations

    def _solve(self, inputs: "Inputs") -> Solution:
        """
        Run the EIS simulation to calculate impedance at all specified frequencies.

        Parameters
        ----------
        inputs : Inputs
            Input parameters.
        calculate_sensitivities : bool
            Whether to calculate sensitivities (default: False).
            Currently not implemented for EIS.

        Returns
        -------
        Solution
            Complex impedance results.
        """
        # Always run initialise_eis_matrices, after rebuilding the model if necessary
        self._model_rebuild(inputs)

        solution = Solution()
        if self.time_data is None:
            ## Stationary EIS
            zs = [self.calculate_impedance(frequency) for frequency in self._f_eval]
            solution.set_solution_variable(
                "Impedance", data=np.asarray(zs) * self.z_scale
            )

        else:
            ## Operando EIS
            zs_at_time_t = []
            for i in range(len(self.time_data)):
                self._jac = self._jac_at_time_t[i]
                zs = [self.calculate_impedance(frequency) for frequency in self._f_eval]
                zs_at_time_t.append(zs)
            solution.set_solution_variable("Time [s]", data=np.asarray(self.time_data))
            solution.set_solution_variable(
                "Impedance", data=np.asarray(zs_at_time_t) * self.z_scale
            )

        return solution

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
        A = 1.0j * 2 * np.pi * frequency * self._mass - self._jac

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
    def time_data(self):
        return self._simulation.time_data

    @property
    def has_sensitivities(self):
        return False

    def copy(self):
        """Return a copy of the simulation."""
        return copy(self)
