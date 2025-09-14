from copy import deepcopy

import numpy as np
import pybamm

from pybop import FailedSolution, Inputs, RecommendedSolver


class RebuildableSimulation:
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
    cost_names : list[str]
        A list of the cost variable names.
    input_parameter_names : list[str], optional
        A list of the input parameter names.
    parameter_values : pybamm.ParameterValues, optional
    initial_state : float | str, optional
        The initial state of charge or voltage for the battery model. If float, it will be
        represented as SoC and must be in range 0 to 1. If str, it will be represented as voltage and
        needs to be in the format: "3.4 V".
        The parameter values to be used in the model.
    t_eval : np.ndarray, optional
        The time points to stop the solver at. These points should be used to inform the solver of
        discontinuities in the solution.
    t_interp : np.ndarray, optional
        The time points at which to interpolate the solution. If None, no interpolation will be done.
    experiment : pybamm.Experiment | string | list, optional
        The experimental conditions under which to solve the model. If a string is passed, the
        experiment is constructed as `pybamm.Experiment([experiment])`. If a list is passed, the
        experiment is constructed as `pybamm.Experiment(experiment)`.
    solver : pybamm.BaseSolver, optional
        The solver to use to solve the model. If None, uses `pybop.RecommendedSolver`.
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
        cost_names: list[str] = None,
        input_parameter_names: list[str] | None = None,
        parameter_values: pybamm.ParameterValues | None = None,
        initial_state: float | str | None = None,
        t_eval: np.ndarray | None = None,
        t_interp: np.ndarray | None = None,
        experiment: pybamm.Experiment | None = None,
        solver: pybamm.BaseSolver | None = None,
        geometry: pybamm.Geometry | None = None,
        submesh_types: dict | None = None,
        var_pts: dict | None = None,
        spatial_methods: dict | None = None,
        discretisation_kwargs: dict | None = None,
        build_every_time: bool = False,
    ):
        # Core
        self._model = model
        self._cost_names = cost_names
        self._parameter_values = (
            parameter_values.copy()
            if parameter_values is not None
            else model.default_parameter_values
        )

        # Simulation Params
        self._initial_state = initial_state
        self._t_eval = [t_eval[0], t_eval[-1]] if t_eval is not None else None
        self._t_interp = t_interp
        self._experiment = experiment

        # Configuration
        self._solver = solver.copy() if solver is not None else RecommendedSolver()
        self._geometry = geometry or model.default_geometry
        self._submesh_types = submesh_types or model.default_submesh_types
        self._var_pts = var_pts or model.default_var_pts
        self._spatial_methods = spatial_methods or model.default_spatial_methods
        self._discretisation_kwargs = discretisation_kwargs or {"check_model": True}

        # State
        self._built_model = None
        self._sim_experiment = None
        self._solve = None
        self._calculate_sensitivities = False

        # Build
        self._input_parameter_names = input_parameter_names
        self._requires_model_rebuild = self._determine_rebuild_requirement(
            build_every_time
        )
        self._set_up_solution_method()

    def _determine_rebuild_requirement(self, build_every_time: bool | None) -> bool:
        """Determine if model needs rebuilding on each evaluation."""

        # If there are no optimisation parameters, model does not need rebuilding
        if not self._input_parameter_names:
            return False

        # All non-experiment protocols with an initial state require model rebuilding
        if self._experiment is None and self._initial_state is not None:
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

    def _set_up_solution_method(self) -> None:
        """Configure the mode of operation."""

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

            # Speed up the solver with output_variables when not using an experiment
            if self._solver.output_variables == []:
                """DISABLE until PyBaMM PR 5118 is resolved"""
                # self._solver.output_variables = self._cost_names or []
                pass

            # Build if only building once, otherwise build on evalution
            if not self._requires_model_rebuild:
                self.build_model()
                self._solve = self._solve_in_time_without_rebuild
            else:
                self._solve = self._solve_in_time_with_rebuild

    def solve(
        self, inputs: Inputs | list[Inputs], calculate_sensitivities: bool = False
    ) -> list[pybamm.Solution | FailedSolution]:
        """
        Run the simulation using the built model and solver.

        Parameters
        ---------
        calculate_sensitivities : bool
            Whether to calculate sensitivities or not.

        Returns
        -------
        solution : list[pybamm.Solution | pybop.FailedSolution]
            The pybamm solution object.
        """
        # Convert and standardise inputs as a list of candidate dictionaries
        inputs_list = inputs if isinstance(inputs, list) else [inputs]
        self._calculate_sensitivities = calculate_sensitivities

        # Check for expected input parameters
        if set(inputs_list[0].keys()) != set(self._input_parameter_names):
            raise ValueError(
                "The inputs do not contain the expected parameters. "
                f"The inputs keys are {list(inputs_list[0].keys())}, "
                f"but the expected inputs are {self._parameter_names}."
            )

        # The underlying solve method is one of four methods set during initialisation
        solutions = self._solve(inputs_list)

        return self._process_solutions(solutions)

    """ ______ ______ ATTRIBUTES FOR SOLVING IN TIME, WITHOUT AN EXPERIMENT ______ ______  """

    def rebuild_model(self, inputs: Inputs) -> None:
        """Update the parameter values and rebuild the model, if required."""
        if not self._requires_model_rebuild:
            # Parameter values will be passed to the solver as inputs
            return

        # Update the parameter values and build again

        self._parameter_values.update(inputs)
        self.build_model()

    def build_model(self) -> None:
        """Build the model using the given parameter values."""
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
        self, inputs: list[Inputs]
    ) -> list[pybamm.Solution]:
        """Solve in time without rebuilding the PyBaMM model."""
        if len(inputs) == 1:
            return [self._pybamm_solve(inputs=inputs[0])]
        return self._pybamm_solve(inputs=inputs)

    def _solve_in_time_with_rebuild(
        self, inputs: list[Inputs]
    ) -> list[pybamm.Solution]:
        """Solve in time, rebuilding the model for each set of inputs."""
        solutions = []
        for x in inputs:
            self.rebuild_model(x)
            solutions.append(self._pybamm_solve(inputs=None))
        return solutions

    def _pybamm_solve(
        self, inputs: Inputs | list[Inputs] | None
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
        self, inputs: list[Inputs]
    ) -> list[pybamm.Solution]:
        """Simulate an experiment without rebuilding the PyBaMM model."""
        solutions = []
        for x in inputs:
            sol = self._sim_experiment.solve(inputs=x, initial_soc=self._initial_state)
            solutions.append(sol)
        return solutions

    def _simulate_experiment_with_rebuild(
        self, inputs: list[Inputs]
    ) -> list[pybamm.Solution]:
        """Simulate an experiment, rebuilding the simulation for each set of inputs."""
        solutions = []
        for x in inputs:
            # Update parameters and create new simulation
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
                    self._cost_names, list(self._input_parameter_names)
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
    def solver(self):
        return self._solver

    @property
    def requires_model_rebuild(self):
        return self._requires_model_rebuild
