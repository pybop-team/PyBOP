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
from pybop.processing.dataset import Dataset, get_impedance_variables
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
        A dataset defining a time-domain protocol, containing the domain data, a control
        variable (e.g. "Current [A]") and the impedance variables given
        by `pybop.get_impedance_variables(f_eval)`. These are non-zero at the times at
        which a spectrum was measured, and zero elsewhere; the simulator computes a
        spectrum at exactly those times, coupling the time-domain simulation to the
        EIS. If None, a single spectrum is computed about the initial state.
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
    """

    def __init__(
        self,
        model: pybamm.BaseModel,
        f_eval: np.ndarray | list[float],
        parameter_values: pybamm.ParameterValues | None = None,
        protocol: Dataset | None = None,
        initial_state: float | str | None = None,
        solver: pybamm.BaseSolver | None = None,
        geometry: pybamm.Geometry | None = None,
        submesh_types: dict | None = None,
        var_pts: dict | None = None,
        spatial_methods: dict | None = None,
        discretisation_kwargs: dict | None = None,
        cache_esoh: bool = True,
        build_every_time: bool = False,
    ):
        # Set-up model for EIS
        self._f_eval = f_eval
        model = self.set_up_for_eis(model)
        parameter_values = parameter_values or model.default_parameter_values
        parameter_values["Current function [A]"] = 0

        super().__init__(parameters=parameter_values)

        # Locate the times at which to compute a spectrum, if any
        self._impedance_variables = get_impedance_variables(f_eval)
        self._acquisition_indices = self._locate_acquisition_times(protocol)

        # Set up a simulation. When a protocol is given, the Simulator installs the
        # control interpolant and sets t_interp to the domain data, so the entries of
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
        )
        self.debug_mode = False

        # Initialise
        self.M = None
        self.b = None

        v_scale = getattr(model.variables["Voltage [V]"], "scale", 1)
        i_scale = getattr(model.variables["Current [A]"], "scale", 1)
        self.z_scale = self.parameter_values.evaluate(v_scale / i_scale)

    def _locate_acquisition_times(self, protocol: Dataset | None) -> np.ndarray | None:
        """
        Return the indices of the rows at which a spectrum was measured, or None for a
        stationary simulation.
        """
        if protocol is None:
            return None

        missing = set(self._impedance_variables) - set(protocol.keys())
        if missing:
            raise ValueError(
                "The protocol dataset is missing impedance variables, e.g. "
                f"'{sorted(missing)[0]}'. Name them with pybop.get_impedance_variables(f_eval), "
                "using zeros at the times where no spectrum was measured."
            )

        measured = np.asarray([protocol[name] for name in self._impedance_variables])
        indices = np.flatnonzero(np.any(measured != 0.0, axis=0))
        if indices.size == 0:
            raise ValueError(
                "The impedance variables are zero everywhere, so there is nothing to fit. "
                "Set them to the measured spectra at the times of acquisition."
            )
        return indices

    def set_output_variables(self, target: list[str]):
        """
        Deliberately a no-op. Restricting the solver to a list of output variables stops
        PyBaMM from returning the state vector, which is required to linearise the model
        about the state at each acquisition time.
        """
        return None

    def _set_up_matrices(self, inputs: "Inputs") -> None:
        """
        Set up the solver and the parts of the linear system which do not depend on the
        operating point: the mass matrix and the forcing vector. Called once, unless the
        model has to be rebuilt.
        """
        built_model = self.simulation.built_model
        # Sort so the compiled functions expect the same order that
        # BaseSolver._set_up_model_inputs() will use if the time-domain simulation
        # recompiles them in _solve_along_protocol(). _jacobian() stacks sorted to match.
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

    def _calculate_spectrum(self, jac: csc_matrix) -> np.ndarray:
        """Compute the impedance at every frequency for a given Jacobian."""
        return (
            np.asarray([self.calculate_impedance(f, jac) for f in self._f_eval])
            * self.z_scale
        )

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
            A modified copy of the model, ready for EIS simulations.

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

        # Without a differential surface form the model has no double-layer capacitance,
        # so the impedance contains no charge-transfer arc, only the diffusion response
        if model.options.get("surface form") != "differential":
            warnings.warn(
                "The model does not use a differential surface form, so the impedance "
                'will not contain a charge-transfer arc. Pass options={"surface form": '
                '"differential"} to include the double layer.',
                stacklevel=2,
            )

        # Work on a copy, so that the model given by the user is left untouched
        model = model.new_copy()

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
                    simulations.append(
                        FailedSolution(self.solution_variables, x.keys())
                    )
            return simulations

        simulations = []
        for x in inputs:
            simulations.append(self._solve(x))
        return simulations

    def _solve(self, inputs: "Inputs") -> Solution:
        """
        Run the EIS simulation to calculate impedance at all specified frequencies.

        For a stationary simulation, one spectrum is computed about the initial state.
        When coupled to a protocol, the time-domain trajectory is solved once and each
        spectrum is computed by linearising about the state at the requested time.

        Parameters
        ----------
        inputs : Inputs
            Input parameters.

        Returns
        -------
        Solution
            Complex impedance results, or the voltage and the real and imaginary
            impedance components over the time domain when coupled to a protocol.
        """
        if self._acquisition_indices is None:
            # Rebuild the model only if necessary, then set up the constant matrices
            self._model_rebuild(inputs)
            y0 = self.simulation.built_model.concatenated_initial_conditions.evaluate(
                0, inputs=inputs
            )
            solution = Solution()
            solution.set_solution_variable(
                "Impedance",
                data=self._calculate_spectrum(self._jacobian(0, y0, inputs)),
            )
            return solution

        return self._solve_along_protocol(inputs)

    def _solve_along_protocol(self, inputs: "Inputs") -> Solution:
        """
        Solve the time-domain protocol once, then compute a spectrum about the state at
        each of the requested times.
        """
        sim_solution = self._simulator.solve(inputs)
        if isinstance(sim_solution, FailedSolution):
            raise ValueError("The time-domain simulation failed.")

        # Set up the constant matrices only after the time-domain solve, which discards
        # the simulation when the model has to be rebuilt for every set of inputs
        self._model_rebuild(inputs)

        t, y = sim_solution.t, sim_solution.y
        if self._acquisition_indices[-1] >= len(t):
            raise ValueError(
                "The time-domain simulation terminated before the last EIS time."
            )

        solution = Solution()
        solution.set_solution_variable("Time [s]", data=t)
        solution.set_solution_variable(
            "Voltage [V]", data=sim_solution["Voltage [V]"].data
        )

        # Zero away from the times of acquisition, matching the dataset convention
        impedance = {name: np.zeros(len(t)) for name in self._impedance_variables}
        for i in self._acquisition_indices:
            y_i = np.asarray(y[:, i]).reshape(-1, 1)
            if np.abs(y_i[-1]) > 1e-10:
                warnings.warn(
                    f"The current is not zero at the acquisition time t={t[i]} s, "
                    "so the impedance is linearised about a non-zero operating point.",
                    stacklevel=2,
                )
            zs = self._calculate_spectrum(self._jacobian(t[i], y_i, inputs))
            for j, z in enumerate(zs):
                impedance[self._impedance_variables[2 * j]][i] = z.real
                impedance[self._impedance_variables[2 * j + 1]][i] = z.imag

        for name, data in impedance.items():
            solution.set_solution_variable(name, data=data)
        return solution

    def calculate_impedance(self, frequency: float, jac: csc_matrix) -> complex:
        """
        Calculate the impedance for a given frequency.

        This method computes the system matrix, solves the linear system, and calculates
        the impedance based on the solution.

        Parameters
        ----------
        frequency : float
            The frequency at which to calculate the impedance in Hz.
        jac : csc_matrix
            The Jacobian of the built model about the operating point.

        Returns
        -------
        complex
            The calculated impedance.
        """

        # Compute the system matrix
        A = 1.0j * 2 * np.pi * frequency * self.M - jac

        # Solve the system
        x = spsolve(A, self.b)

        # Calculate the impedance (voltage / current)
        return -x[-2] / x[-1]

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
    def solution_variables(self) -> list[str]:
        """The names of the variables set by a solve."""
        if self._acquisition_indices is None:
            return ["Impedance"]
        return ["Time [s]", "Voltage [V]", *self._impedance_variables]

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
