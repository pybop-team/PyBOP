from copy import copy, deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pybamm
from pybamm import SolverError

if TYPE_CHECKING:
    from pybop.parameters.parameter import Inputs
from pybop.parameters.parameter import Parameter
from pybop.processing.dataset import Dataset
from pybop.pybamm.utils import RecommendedSolver
from pybop.simulators.base_simulator import BaseSimulator
from pybop.simulators.failed_solution import FailedSolution


class Simulator(BaseSimulator):
    """
    A class to automatically build/rebuild and solve a pybamm.Simulation for a given model and protocol.

    There are two contexts in which this class can be used:
    1. A pybamm model can be built once and then run multiple times with different inputs.
    2. A pybamm model needs to be built and then run for each set of inputs, for example in the case
        where one of the inputs is a geometric parameter which requires a new mesh.

    The logic for (1) and (2) happens automatically. To override this logic, the argument `build_every_time`
    can be set to `True` which will force (2) to occur.

    Parameters
    ----------
    model : pybamm.BaseModel
        The PyBaMM model to be used.
    parameter_values : pybamm.ParameterValues, optional
        The parameter values to be used in the model.
    initial_state : dict, optional
        A valid initial state, e.g. `"Initial open-circuit voltage [V]"` or ``"Initial SoC"`.
        Defaults to None, indicating that the existing initial state of charge (for an ECM)
        or initial concentrations (for an EChem model) will be used.
    protocol : pybamm.Experiment | Dataset | np.ndarray | None
        The protocol as an experiment, a 1D array of values or dataset containing (time) domain data.
    solver : pybamm.BaseSolver, optional
        The solver to use to solve the model. If None, uses `pybop.pybamm.RecommendedSolver`.
    output_variables : list, optional
        A list of output variables to return.
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
        parameter_values: pybamm.ParameterValues | None = None,
        initial_state: dict | None = None,
        protocol: pybamm.Experiment | Dataset | np.ndarray | None = None,
        solver: pybamm.BaseSolver | None = None,
        output_variables: list[str] | None = None,
        geometry: pybamm.Geometry | None = None,
        submesh_types: dict | None = None,
        var_pts: dict | None = None,
        spatial_methods: dict | None = None,
        discretisation_kwargs: dict | None = None,
        build_every_time: bool = False,
    ):
        # Core
        self._model = model
        self._parameter_values = (
            parameter_values.copy()
            if parameter_values is not None
            else model.default_parameter_values
        )
        self._output_variables = output_variables

        # Replace any PyBaMM InputParameter with a PyBOP Parameter
        for name, param in self._parameter_values.items():
            if isinstance(param, pybamm.InputParameter):
                self._parameter_values[name] = Parameter()

        super().__init__(parameters=self._parameter_values)

        # Simulation params
        self._initial_state = self.convert_to_pybamm_initial_state(initial_state)
        self._experiment = None
        self._t_eval = None
        self._t_interp = None
        self._set_protocol(protocol=protocol)

        # Solver set-up
        if solver is not None:
            solver = solver.copy()
        elif isinstance(self._model.default_solver, pybamm.DummySolver):
            solver = self._model.default_solver
        else:
            solver = RecommendedSolver()
        self._solver = solver
        if not self._solver.supports_interp:
            self._t_eval = self._t_interp
            self._t_interp = None

        # Configuration
        self._geometry = geometry or model.default_geometry
        self._submesh_types = submesh_types or model.default_submesh_types
        self._var_pts = var_pts or model.default_var_pts
        self._spatial_methods = spatial_methods or model.default_spatial_methods
        self._discretisation_kwargs = discretisation_kwargs or {"check_model": True}

        # State
        self._simulation = None
        self._solve = None
        self.debug_mode = False
        self.verbose = False

        # Build
        self._input_parameter_names = self.parameters.names
        self._requires_model_rebuild = self._determine_rebuild_requirement(
            build_every_time
        )
        self._set_up_solution_method(output_variables=output_variables)

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all instance attributes.
        # Copy() method avoid modifying the original state.
        state = self.__dict__.copy()

        # Remove the unpicklable entries.
        del state["_simulation"]
        del state["_solve"]
        return state

    def __setstate__(self, state):
        # Restore instance attributes.
        self.__dict__.update(state)

        # Restore unpickalable attributes
        self._simulation = None
        self._solve = None

        self._set_up_solution_method(output_variables=self.output_variables)

    def _set_protocol(self, protocol: pybamm.Experiment | Dataset | np.ndarray | None):
        """
        Set up the protocol for the simulation.

        Parameters
        ----------
        protocol : pybamm.Experiment | pybop.Dataset | np.ndarray | None
            The protocol as an experiment, a 1D array of values or dataset containing (time) domain data.

        Attributes
        ----------
        t_eval : np.ndarray, optional
            The time points to stop the solver at. These points should be used to inform the solver of
            discontinuities in the solution.
        t_interp : np.ndarray, optional
            The time points at which to interpolate the solution. If None, no interpolation will be done.
        experiment : pybamm.Experiment | string | list, optional
            The experimental conditions under which to solve the model.
        """
        if protocol is None:
            self._experiment = None
            self._t_eval = None
            self._t_interp = None
        elif isinstance(protocol, pybamm.Experiment):
            self._experiment = protocol
            self._t_eval = None
            self._t_interp = None
        elif isinstance(protocol, Dataset):
            self._experiment = None
            time_data = protocol[protocol.domain]
            self._t_eval = [time_data[0], time_data[-1]]
            self._t_interp = time_data
            for key in protocol.control_functions:
                control = key.replace(" function", "")
                self._parameter_values[key] = protocol.get_interpolant(control)
        else:
            self._experiment = None
            time_data = protocol
            self._t_eval = [time_data[0], time_data[-1]]
            self._t_interp = time_data
        # else:
        #     raise ValueError(f"Expected an experiment or a dataset. Received {type(protocol)}")

    def _determine_rebuild_requirement(self, build_every_time: bool | None) -> bool:
        """Determine if model needs rebuilding on each evaluation."""

        # If there are no optimisation parameters, model does not need rebuilding
        if not self._input_parameter_names:
            return False

        # If the model builds successfully with empty inputs, it does not need rebuilding
        try:
            self.create_simulation()
            self._simulation.build(initial_soc=self._initial_state, inputs=None)
            requires_model_rebuild = False
        except (ValueError, TypeError, IndexError):
            requires_model_rebuild = True

        # Reset
        self._simulation = None

        if requires_model_rebuild or build_every_time:
            return True
        return False

    def convert_to_pybamm_initial_state(self, initial_state: dict | None) -> None:
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

    def _set_up_solution_method(
        self, output_variables: list[str] | None = None
    ) -> None:
        """Configure the mode of operation."""
        if self._experiment is None:
            # Speed up the solver with output_variables if provided
            self._solver.output_variables = output_variables or []

            # Remove all voltage-based events when not using an experiment
            self._model.events = [e for e in self._model.events if "[V]" not in e.name]

        # Build if only building once, otherwise build on evaluation
        if not self._requires_model_rebuild:
            self.create_simulation()
            if self._experiment is None:
                self._simulation.build(initial_soc=self._initial_state)
            self._solve = self._simulate_without_rebuild
        else:
            self._solve = self._simulate_with_rebuild

    def solve(
        self,
        inputs: "Inputs | list[Inputs] | None" = None,
        calculate_sensitivities: bool = False,
    ) -> pybamm.Solution | FailedSolution | list[pybamm.Solution | FailedSolution]:
        """
        Run the simulation using the built model and solver for one or more sets of
        inputs and return the solution object(s).

        Parameters
        ----------
        inputs : Inputs | list[Inputs], optional
            Input parameters (default: None).
        calculate_sensitivities : bool
            Whether to also return the sensitivities (default: False).

        Returns
        -------
        pybamm.Solution | pybop.FailedSolution | list[pybamm.Solution | pybop.FailedSolution]
            The solution object(s).
        """
        # Convert and standardise inputs as a list of candidate dictionaries
        inputs = inputs or {}
        if not isinstance(inputs, list):
            return self.solve_batch([inputs], calculate_sensitivities)[0]

        return self.solve_batch(inputs, calculate_sensitivities)

    def solve_batch(
        self,
        inputs: "list[Inputs]",
        calculate_sensitivities: bool = False,
    ) -> list[pybamm.Solution | FailedSolution]:
        """
        Run the simulation using the built model and solver for each set of inputs
        and return dict-like simulation results.

        Parameters
        ----------
        inputs : list[Inputs]
            A list of input parameters.
        calculate_sensitivities : bool
            Whether to also return the sensitivities (default: False).

        Returns
        -------
        list[pybamm.Solution | pybop.FailedSolution]
            A list of len(inputs) containing the solution objects.
        """
        if inputs == []:
            return []

        # Check for expected input parameters
        if set(inputs[0].keys()) != set(self._input_parameter_names):
            raise ValueError(
                "The inputs do not contain the expected parameters. "
                f"The inputs keys are {list(inputs[0].keys())}, "
                f"but the expected inputs are {self._input_parameter_names}."
            )

        # Set whether to compute the sensitivities
        if calculate_sensitivities and not self.has_sensitivities:
            raise ValueError("Sensitivities are not available.")

        # Gather solve options
        options = {
            "initial_soc": self._initial_state,
            "calculate_sensitivities": calculate_sensitivities,
        }
        if self._experiment is None:
            options.update({"t_eval": self._t_eval, "t_interp": self._t_interp})

        # The underlying solve method is one of two methods set during initialisation
        return self._process_solutions(self._catch_errors(inputs, options))

    def _catch_errors(self, inputs: "list[Inputs]", options: dict):
        if not self.debug_mode:
            try:
                return self._solve(inputs, options)

            except (
                SolverError,
                ZeroDivisionError,
                RuntimeError,
                ValueError,
                Exception,
            ):
                # Try separately
                solutions = []
                for x in inputs:
                    try:
                        solutions += self._solve([x], options)
                    except (
                        SolverError,
                        ZeroDivisionError,
                        RuntimeError,
                        ValueError,
                        Exception,
                    ) as e:
                        if self.verbose:
                            print(f"Ignoring this sample due to: {e}")
                        solutions.append(
                            FailedSolution(
                                self.output_variables, self._input_parameter_names
                            )
                        )
                return solutions

        return self._solve(inputs, options)

    def create_simulation(self) -> pybamm.Simulation:
        """Create a simulation with current configuration and optional experiment."""
        self._simulation = pybamm.Simulation(
            self._model.new_copy(),
            parameter_values=self._parameter_values,
            experiment=self._experiment,
            solver=self._solver.copy(),
            geometry=deepcopy(self._geometry),
            submesh_types=self._submesh_types,
            var_pts=self._var_pts,
            spatial_methods=self._spatial_methods,
            discretisation_kwargs=self._discretisation_kwargs,
        )

    def _simulate_without_rebuild(
        self, inputs: "list[Inputs]", options: dict
    ) -> list[pybamm.Solution]:
        """Simulate without rebuilding the PyBaMM model."""
        if len(inputs) == 1:
            return [self._simulation.solve(inputs=inputs[0], **options)]

        return self._simulation.solve(inputs=inputs, **options)

    def _simulate_with_rebuild(
        self, inputs: "list[Inputs]", options: dict
    ) -> list[pybamm.Solution]:
        """Simulate, rebuilding the simulation for each set of inputs."""
        unmodified_parameter_values = self._parameter_values.copy()

        solutions = []
        for x in inputs:
            # Update parameters and create new simulation
            self._parameter_values.update(x)
            self.create_simulation()
            solutions.append(self._simulation.solve(**options))

        self._simulation = None
        self._parameter_values = unmodified_parameter_values
        return solutions

    def _process_solutions(
        self, solutions: list[pybamm.Solution]
    ) -> list[pybamm.Solution | FailedSolution]:
        """Convert failed solutions to FailedSolution objects."""
        processed_solutions = []
        for solution in solutions:
            if hasattr(solution, "termination") and solution.termination == "failure":
                failed_solution = FailedSolution(
                    self.output_variables, self._input_parameter_names
                )
                processed_solutions.append(failed_solution)
            else:
                processed_solutions.append(solution)
        return processed_solutions

    @property
    def built_model(self):
        return None if self._simulation is None else self._simulation._built_model  # noqa: SLF001

    @property
    def input_parameter_names(self):
        return self._input_parameter_names

    @property
    def model(self):
        return self._model

    @property
    def parameter_values(self):
        return self._parameter_values

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def experiment(self):
        return self._experiment

    @property
    def solver(self):
        return self._solver

    @property
    def output_variables(self):
        return self._output_variables

    def set_output_variables(self, value: list[str] | None):
        if value is not None:
            for var in value:
                if var not in self._model.variable_names():
                    raise ValueError(f"{var} is not a variable in the model.")

        self._output_variables = value
        if self.experiment is None:
            self._set_up_solution_method(output_variables=value)

    @property
    def requires_model_rebuild(self):
        return self._requires_model_rebuild

    @property
    def has_sensitivities(self):
        if (
            self._initial_state is not None
            or self._experiment is not None
            or not self._solver.supports_interp
        ):
            return False
        return True

    def copy(self):
        """Return a copy of the simulation."""
        return copy(self)
