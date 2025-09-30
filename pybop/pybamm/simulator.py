import warnings
from copy import copy, deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pybamm
from pybamm import SolverError

if TYPE_CHECKING:
    from pybop.parameters.parameter import Inputs
from pybop._dataset import Dataset
from pybop._utils import FailedSolution, RecommendedSolver
from pybop.pybamm.parameter_utils import set_formation_concentrations
from pybop.simulators.base_simulator import BaseSimulator


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
    input_parameter_names : list[str], optional
        A list of the input parameter names.
    initial_state : dict, optional
        A valid initial state, e.g. `"Initial open-circuit voltage [V]"` or ``"Initial SoC"`.
        Defaults to None, indicating that the existing initial state of charge (for an ECM)
        or initial concentrations (for an EChem model) will be used.
    protocol : pybamm.Experiment | Dataset | np.ndarray | None
        The protocol as an experiment, a 1D array of values or dataset containing (time) domain data.
    solver : pybamm.BaseSolver, optional
        The solver to use to solve the model. If None, uses `pybop.RecommendedSolver`.
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
    use_formation_concentrations : bool, optional
        If True and "Initial concentration in negative electrode [mol.m-3]" is in the parameter set,
        the total quantity of lithium will be moved to the positive electrode prior to applying any
        inputs or initial state (default: False).
    """

    def __init__(
        self,
        model: pybamm.BaseModel,
        parameter_values: pybamm.ParameterValues | None = None,
        input_parameter_names: str | list[str] | None = None,
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
        use_formation_concentrations: bool = False,
    ):
        # Core
        self._model = model
        self._parameter_values = (
            parameter_values.copy()
            if parameter_values is not None
            else model.default_parameter_values
        )
        self._output_variables = output_variables
        self.use_formation_concentrations = use_formation_concentrations

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

        # Warnings
        self.exception = [
            "These parameter values are infeasible."
        ]  # TODO: Update to a utility function and add to it on exception creation
        self.warning_patterns = [
            "Ah is greater than",
            "Non-physical point encountered",
        ]
        self.debug_mode = False
        self.verbose = False

        # State
        self._built_model = None
        self._sim_experiment = None
        self._solve = None
        self._calculate_sensitivities = False

        # Build
        input_names = input_parameter_names or []
        self._input_parameter_names = (
            input_names if isinstance(input_names, list | None) else [input_names]
        )
        self._requires_model_rebuild = self._determine_rebuild_requirement(
            build_every_time
        )
        self._set_up_solution_method(output_variables=output_variables)

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
            control = "Current function [A]"
            if control in protocol.data.keys():
                self._parameter_values[control] = pybamm.Interpolant(
                    protocol["Time [s]"],
                    protocol[control],
                    pybamm.t,
                )
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

        # All non-experiment protocols with an initial state require model rebuilding
        if self._experiment is None and self._initial_state is not None:
            return True

        # All protocols which require resetting to formation conditions require rebuiding
        if self.use_formation_concentrations:
            return True

        # Test whether the model needs rebuilding by marking parameters as inputs
        unmodified_parameter_values = self._parameter_values.copy()
        for param in self._input_parameter_names:
            self._parameter_values.update({param: "[input]"})

        # If the model builds successfully with inputs, it does not need rebuilding
        try:
            self.build_model()
            requires_model_rebuild = False
        except (ValueError, TypeError):
            self._built_model = None
            requires_model_rebuild = True

        if requires_model_rebuild or build_every_time:
            self._parameter_values = unmodified_parameter_values  # reset
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

        # Speed up the solver with output_variables if provided
        self._solver.output_variables = output_variables or []

        if self._experiment is not None:
            # Build if only building once, otherwise build on evalution
            if not self._requires_model_rebuild:
                self._sim_experiment = self._create_experiment_simulation()
                self._solve = self._simulate_experiment_without_rebuild
            else:
                self._solve = self._simulate_experiment_with_rebuild

        else:
            # Remove all voltage-based events when not using an experiment
            self._model.events = [e for e in self._model.events if "[V]" not in e.name]

            # Build if only building once, otherwise build on evalution
            if not self._requires_model_rebuild:
                self.build_model()
                self._solve = self._solve_in_time_without_rebuild
            else:
                self._solve = self._solve_in_time_with_rebuild

    def simulate(
        self,
        inputs: "Inputs | None" = None,
        calculate_sensitivities: bool = False,
    ) -> (
        dict[str, np.ndarray]
        | tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]
    ):
        sol = self.solve(inputs=inputs, calculate_sensitivities=calculate_sensitivities)

        if calculate_sensitivities:
            return (
                {s: sol[s].data for s in self.output_variables},
                {
                    p: {
                        s: np.asarray(sol[s].sensitivities[p])
                        for s in self.output_variables
                    }
                    for p in self.parameters.keys()
                },
            )

        return {s: sol[s].data for s in self.output_variables}

    def solve(
        self,
        inputs: "Inputs | None" = None,
        calculate_sensitivities: bool = False,
    ) -> list[pybamm.Solution | FailedSolution]:
        """
        Run the simulation using the built model and solver.

        Parameters
        ---------
        calculate_sensitivities : bool
            Whether to calculate sensitivities or not.

        Returns
        -------
        list[pybamm.Solution | pybop.FailedSolution] | pybamm.Solution | pybop.FailedSolution
            A list of solution objects or one solution object.
        """
        # Convert and standardise inputs as a list of candidate dictionaries
        inputs = inputs or {}
        if isinstance(inputs, list):
            inputs_list = inputs
            return_as_list = True
        else:
            inputs_list = [inputs]
            return_as_list = False

        # Check for expected input parameters
        if set(inputs_list[0].keys()) != set(self._input_parameter_names):
            raise ValueError(
                "The inputs do not contain the expected parameters. "
                f"The inputs keys are {list(inputs_list[0].keys())}, "
                f"but the expected inputs are {self._input_parameter_names}."
            )

        # Set whether to compute the sensitivities
        if calculate_sensitivities and not self.has_sensitivities:
            raise ValueError("Sensitivities are not available.")
        self._calculate_sensitivities = calculate_sensitivities

        # The underlying solve method is one of four methods set during initialisation
        solutions = self._process_solutions(self._catch_errors(inputs_list))

        if return_as_list:
            return solutions
        return solutions[0]

    def _catch_errors(self, inputs_list: "list[Inputs]"):
        if not self.debug_mode:
            with warnings.catch_warnings():
                for pattern in self.warning_patterns:
                    warnings.filterwarnings(
                        "error", category=UserWarning, message=pattern
                    )

                try:
                    return self._solve(inputs_list)
                except (SolverError, ZeroDivisionError, RuntimeError, ValueError) as e:
                    if isinstance(e, ValueError) and str(e) not in self.exception:
                        raise  # Raise the error if it doesn't match the expected list
                    return [
                        FailedSolution(
                            self.output_variables, self._input_parameter_names
                        )
                    ]
                except (UserWarning, Exception) as e:
                    if self.verbose:
                        print(f"Ignoring this sample due to: {e}")
                    return [
                        FailedSolution(
                            self.output_variables, self._input_parameter_names
                        )
                    ]

        return self._solve(inputs_list)

    """ ______ ______ ATTRIBUTES FOR SOLVING IN TIME, WITHOUT AN EXPERIMENT ______ ______  """

    def rebuild_model(self, inputs: "Inputs") -> None:
        """Update the parameter values and rebuild the model, if required."""
        if not self._requires_model_rebuild:
            # Parameter values will be passed to the solver as inputs
            return

        # Update the parameter values and build again
        if self.use_formation_concentrations:
            set_formation_concentrations(self._parameter_values)
        self._parameter_values.update(inputs)
        self.build_model()

    def build_model(self) -> None:
        """Build the model using the given parameter values."""
        # Build pybamm model if not already built
        if not self._model.built:
            self._model.build_model()

        model = self._model.new_copy()
        geometry = deepcopy(self._geometry)

        if self._experiment is None and self._initial_state is not None:
            self._parameter_values.set_initial_state(
                self._initial_state, param=model.param, options=model.options
            )

        self._parameter_values.process_geometry(geometry)
        self._parameter_values.process_model(model)

        mesh = pybamm.Mesh(geometry, self._submesh_types, self._var_pts)
        disc = pybamm.Discretisation(
            mesh, self._spatial_methods, **self._discretisation_kwargs
        )
        disc.process_model(model)

        self._built_model = model
        self._solver = self._solver.copy()  # reset solver for new model

    def _solve_in_time_without_rebuild(
        self, inputs: "list[Inputs]"
    ) -> list[pybamm.Solution]:
        """Solve in time without rebuilding the PyBaMM model."""
        if len(inputs) == 1:
            return [self._pybamm_solve(inputs=inputs[0])]
        return self._pybamm_solve(inputs=inputs)

    def _solve_in_time_with_rebuild(
        self, inputs: "list[Inputs]"
    ) -> list[pybamm.Solution]:
        """Solve in time, rebuilding the model for each set of inputs."""
        solutions = []
        for x in inputs:
            self.rebuild_model(x)
            solutions.append(self._pybamm_solve(inputs=None))
        return solutions

    def _pybamm_solve(
        self, inputs: "Inputs | list[Inputs] | None"
    ) -> pybamm.Solution | list[pybamm.Solution]:
        """A function that runs the simulation using the built model."""
        return self._solver.solve(
            model=self._built_model,
            inputs=inputs,
            t_eval=self._t_eval,
            t_interp=self._t_interp,
            calculate_sensitivities=self._calculate_sensitivities,
        )

    """ ______ ______ ______ ATTRIBUTES FOR SIMULATING AN EXPERIMENT ______ ______ ______  """

    def _create_experiment_simulation(self) -> pybamm.Simulation:
        """Create a simulation with current configuration and an experiment."""
        return pybamm.Simulation(
            self._model,
            parameter_values=self._parameter_values,
            experiment=self._experiment,
            solver=self._solver,
            geometry=self._geometry,
            submesh_types=self._submesh_types,
            var_pts=self._var_pts,
            spatial_methods=self._spatial_methods,
            discretisation_kwargs=self._discretisation_kwargs,
        )

    def _simulate_experiment_without_rebuild(
        self, inputs: "list[Inputs]"
    ) -> list[pybamm.Solution]:
        """Simulate an experiment without rebuilding the PyBaMM model."""
        solutions = []
        for x in inputs:
            sol = self._sim_experiment.solve(inputs=x, initial_soc=self._initial_state)
            solutions.append(sol)
        return solutions

    def _simulate_experiment_with_rebuild(
        self, inputs: "list[Inputs]"
    ) -> list[pybamm.Solution]:
        """Simulate an experiment, rebuilding the simulation for each set of inputs."""
        solutions = []
        for x in inputs:
            # Update parameters and create new simulation
            if self.use_formation_concentrations:
                set_formation_concentrations(self._parameter_values)
            self._parameter_values.update(x)
            sim = self._create_experiment_simulation()
            solutions.append(sim.solve(initial_soc=self._initial_state))
        return solutions

    """ ______ ______ ______ ______ GENERAL ATTRIBUTES ______ ______ ______ ______ ______  """

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
        return self._built_model

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
