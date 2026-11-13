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
from pybop.processing.dataset import Dataset
from pybop.pybamm.simulator import Simulator
from pybop.pybamm.utils import SymbolReplacer
from pybop.simulators.base_simulator import BaseSimulator, Solution
from pybop.simulators.failed_solution import FailedSolution


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
    protocol : pybop.Dataset, optional
        A dataset defining a time-domain protocol. If None, stationary EIS is computed about the
        initial state.
    acquisition_indices : array-like, optional
        The indices at which to compute a spectrum, required when a Dataset is passed as the protocol.
    initial_state : dict, optional
        A valid initial state, e.g. `"Initial open-circuit voltage [V]"` or ``"Initial SoC"`.
        Defaults to None, indicating that the existing initial state of charge (for an ECM)
        or initial concentrations (for an EChem model) will be used.
    solver : pybamm.BaseSolver, optional
        The solver to simulate the composed Simulator. If None, uses `pybop.pybamm.RecommendedSolver`.
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
    cache_esoh : bool, optional
        If True, the electrode SOH computation is cached for repeated calls to `pybamm.Simulation.solve`
        (default: True).
    build_every_time : bool, optional
        If True, the model will be rebuilt every evaluation. Otherwise, the need to rebuild will be
        determined automatically.
    remove_voltage_events : bool, optional
        If True, the minimum and maximum voltage events are removed (default: True).
    """

    def __init__(
        self,
        model: pybamm.BaseModel,
        f_eval: np.ndarray | list[float],
        parameter_values: pybamm.ParameterValues | None = None,
        protocol: Dataset | None = None,
        acquisition_indices: np.ndarray | list[float] | None = None,
        initial_state: float | str | None = None,
        solver: pybamm.BaseSolver | None = None,
        geometry: pybamm.Geometry | None = None,
        submesh_types: dict | None = None,
        var_pts: dict | None = None,
        spatial_methods: dict | None = None,
        discretisation_kwargs: dict | None = None,
        cache_esoh: bool = True,
        build_every_time: bool = False,
        remove_voltage_events: bool = True,
    ):
        # Locate the times at which to compute a spectrum
        if protocol is None and acquisition_indices is not None:
            raise ValueError(
                "A protocol is required when a list of acquisition_indices is passed."
            )
        elif protocol is not None and acquisition_indices is None:
            warnings.warn(
                "A protocol was not provided so acquisition_indices will not be used.",
                stacklevel=2,
            )
            acquisition_indices = None
        elif acquisition_indices is not None and np.max(acquisition_indices) >= len(
            protocol
        ):
            raise ValueError(
                "One or more acquisition_indices is greater than the length of protocol."
            )
        self._acquisition_indices = acquisition_indices

        # Set-up model for EIS
        self._f_eval = f_eval
        model = self.set_up_for_eis(model)
        parameter_values = (
            parameter_values.copy()
            if parameter_values is not None
            else model.default_parameter_values
        )
        parameter_values["Current function [A]"] = 0

        super().__init__(parameters=parameter_values)

        # Set up a simulation. When a protocol is given, the Simulator overwrites the
        # "Current function [A]" and sets t_interp to the domain data, so the entries of
        # the solution align with the rows of the dataset.
        self._simulator = Simulator(
            model,
            parameter_values=parameter_values,
            protocol=protocol,
            initial_state=initial_state,
            solver=solver,
            geometry=geometry,
            submesh_types=submesh_types,
            var_pts=var_pts,
            spatial_methods=spatial_methods,
            discretisation_kwargs=discretisation_kwargs,
            cache_esoh=cache_esoh,
            build_every_time=build_every_time,
            remove_voltage_events=remove_voltage_events,
        )
        self.debug_mode = False

        # Initialise
        self.M = None
        self.b = None

        v_scale = getattr(model.variables["Voltage [V]"], "scale", 1)
        i_scale = getattr(model.variables["Current [A]"], "scale", 1)
        self._z_scale = self.parameter_values.evaluate(v_scale / i_scale)

        # Set up solution method
        if self._acquisition_indices is None:
            self._solve = self._simulate_stationary
        else:
            self._solve = self._simulate_with_protocol

    def set_output_variables(self, target: list[str]) -> None:
        """
        Deliberately a no-op. Restricting the solver to a list of output variables stops
        PyBaMM from returning the state vector, which is required to linearise the model
        about the state at each acquisition time.
        """
        return None

    def _set_up_matrices(self, inputs: "Inputs") -> None:
        """
        Set up the solver and the parts of the linear system that do not depend on the
        operating point. Called once, unless the model has to be rebuilt.
        """
        built_model = self.simulation.built_model
        # Sort so the compiled functions expect the same order that
        # BaseSolver._set_up_model_inputs() will use if the time-domain simulation
        # recompiles them in _simulate_with_protocol(). _jacobian() stacks sorted to match.
        self.simulation.solver.set_up(
            built_model,
            inputs={k: inputs[k] for k in sorted(inputs)} if inputs else inputs,
        )

        self.M = csc_matrix(built_model.mass_matrix.entries)

        # Add forcing to the RHS on the current density
        self.b = np.zeros((self.M.shape[0], 1))
        self.b[-1] = -1

    def _jacobian(self, t: float, y: np.ndarray, inputs: "Inputs") -> csc_matrix:
        """Evaluate the Jacobian of the built model about the state y at time t."""
        built_model = self.simulation.built_model

        # Stacked in sorted order to match _set_up_matrices() and pybamm's
        # BaseSolver._set_up_model_inputs() convention.
        casadi_inputs = (
            casadi.vertcat(*[inputs[name] for name in sorted(inputs)])
            if inputs is not None and built_model.convert_to_format == "casadi"
            else inputs or []
        )
        jac = built_model.jac_rhs_algebraic_eval(t, y, casadi_inputs).sparse()
        return csc_matrix(jac)

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
        """
        Rebuild the EIS model if required, and set up the operating-point-independent
        matrices. Mirroring the Simulator, the model and these matrices are set up once
        unless a rebuild is required on every evaluation.
        """
        if self._simulator.requires_model_rebuild:
            self.parameter_values.update(inputs)
            self._simulator.create_simulation()
            self.simulation.build(initial_soc=self._simulator.initial_state)
            self._set_up_matrices(inputs=inputs)
        elif self.M is None:
            self._set_up_matrices(inputs=inputs)

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

    def solve_batch(
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
                except (ZeroDivisionError, RuntimeError, ValueError):
                    simulations.append(FailedSolution(self.default_variables, x.keys()))
            return simulations

        simulations = []
        for x in inputs:
            simulations.append(self._solve(x))
        return simulations

    def _simulate_stationary(self, inputs: "Inputs") -> Solution:
        """
        Run the EIS simulation to calculate impedance at all specified frequencies.
        For a stationary simulation, one spectrum is computed about the initial state.

        Parameters
        ----------
        inputs : Inputs
            Input parameters.

        Returns
        -------
        Solution
            Complex impedance results.
        """
        # Rebuild the model only if necessary, then set up the constant matrices
        self._model_rebuild(inputs)

        y0 = self.simulation.built_model.concatenated_initial_conditions.evaluate(
            0, inputs=inputs
        )
        zs = self._calculate_impedance(self._jacobian(0, y0, inputs))

        solution = Solution(inputs)
        solution.set_solution_variable("Impedance [Ohm]", data=zs)
        return solution

    def _simulate_with_protocol(self, inputs: "Inputs") -> Solution:
        """
        Run the EIS simulation to calculate impedance at all specified frequencies.
        When coupled to a protocol, the time-domain trajectory is solved once and then
        a spectrum is computed by linearising about the state at each acquisition time.

        Parameters
        ----------
        inputs : Inputs
            Input parameters.

        Returns
        -------
        Solution
            Complex impedance results, and also the time-domain variables when coupled
            to a protocol.
        """
        protocol_solution = self._simulator.solve(inputs)
        if isinstance(protocol_solution, FailedSolution):
            raise ValueError("The time-domain simulation failed.")

        t, y = protocol_solution.t, protocol_solution.y
        if self._acquisition_indices[-1] >= len(t):
            raise ValueError(
                "The time-domain simulation terminated before the last EIS time."
            )

        # Rebuild the model only if necessary, then set up the constant matrices
        # Note that this must be after any time-domain solve, which discards the
        # simulation when the model has to be rebuilt for every set of inputs
        self._model_rebuild(inputs)

        # Evaluate the impedance at the acquisition times only
        zs = np.empty((len(t), len(self._f_eval)), dtype=np.complex128)
        zs[:] = np.nan
        for i in self._acquisition_indices:
            y_i = np.asarray(y[:, i]).reshape(-1, 1)
            if np.abs(y_i[-1]) > 1e-10:
                warnings.warn(
                    f"The current is not zero at time t={t[i]} s, meaning the "
                    "impedance is linearised about a non-stationary operating point.",
                    stacklevel=2,
                )
            zs[i, :] = self._calculate_impedance(self._jacobian(t[i], y_i, inputs))

        solution = Solution(inputs)
        solution.set_solution_variable("Time [s]", data=t)
        solution.set_solution_variable(
            "Current [A]", data=protocol_solution["Current [A]"].data
        )
        solution.set_solution_variable(
            "Voltage [V]", data=protocol_solution["Voltage [V]"].data
        )
        solution.set_solution_variable("Impedance [Ohm]", data=zs)
        return solution

    def _calculate_impedance(self, jac: csc_matrix) -> np.ndarray:
        """
        Calculate the impedance at every frequency for a given Jacobian.

        This method computes the system matrix, solves the linear system, and calculates
        the impedance based on the solution.

        Parameters
        ----------
        jac : csc_matrix
            The Jacobian of the built model about the operating point.

        Returns
        -------
        np.ndarray
            An array of complex impedance values.
        """
        zs = np.zeros_like(self._f_eval, dtype=np.complex128)
        for i, frequency in enumerate(self._f_eval):
            # Compute the system matrix
            A = 1.0j * 2 * np.pi * frequency * self.M - jac

            # Solve the system
            x = spsolve(A, self.b)

            # Calculate the impedance (voltage / current)
            zs[i] = -x[-2] / x[-1]

        return zs * self._z_scale

    @property
    def simulation(self):
        return self._simulator._simulation  # noqa: SLF001

    @property
    def parameter_values(self):
        return self._simulator.parameter_values

    @property
    def input_parameter_names(self):
        return self._simulator.input_parameter_names

    @property
    def default_variables(self):
        variables = ["Impedance [Ohm]"]
        if self._acquisition_indices is not None:
            variables.extend(self._simulator.default_variables)
        return variables

    @property
    def acquisition_indices(self):
        return self._acquisition_indices

    @property
    def has_sensitivities(self):
        return False

    @property
    def debug_mode(self):
        return self._debug_mode

    @debug_mode.setter
    def debug_mode(self, value: bool):
        self._debug_mode = value
        self._simulator.debug_mode = value

    def copy(self):
        """Return a copy of the simulation."""
        return copy(self)
