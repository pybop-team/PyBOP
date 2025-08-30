from copy import copy, deepcopy
from enum import Enum

import numpy as np
import pybamm

from pybop import FailedSolution, Inputs, Parameters, RecommendedSolver


class OperatingMode(Enum):
    """Enum-like class for operating modes."""

    WITHOUT_EXPERIMENT = "without experiment"
    WITH_EXPERIMENT = "with experiment"


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
        cost_names: list[str] = None,
        pybop_parameters: Parameters | None = None,
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
        build_on_eval: bool | None = None,
    ):
        """
        Parameters
        ---------
        model : pybamm.BaseModel
            The PyBaMM model to be used.
        cost_names : list[str]
            A list of the cost variable names.
        pybop_parameters : pybop.Parameters
            The parameters to optimise.
        parameter_values : pybamm.ParameterValues, optional
            The parameters to be used in the model.
        initial_state: float | str, optional
            The initial state of charge or voltage for the battery model. If float, it will be
            represented as SoC and must be in range 0 to 1. If str, it will be represented as voltage and
            needs to be in the format: "3.4 V".
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
        discretisation_kwargs : dict (optional)
            Any keyword arguments to pass to the Discretisation class.
            See :class:`pybamm.Discretisation` for details.
        build_on_eval : bool
            Boolean to determine if the model will be rebuilt every evaluation. If `initial_state` is
            provided, the model will be rebuilt every evaluation unless `build_on_eval` is `False`, in
            which case the model is built with the parameter values from construction only.
        """
        # Core
        self._model = model
        self._cost_names = cost_names
        self._pybop_parameters = pybop_parameters or Parameters([])
        self._parameter_names = self._pybop_parameters.keys()
        self._parameter_values = parameter_values or model.default_parameter_values

        # Simulation Params
        self.initial_state = initial_state
        self._t_eval = [t_eval[0], t_eval[-1]] if t_eval is not None else None
        self._t_interp = t_interp

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

        # Setup
        self.requires_rebuild = self._determine_rebuild_requirement(build_on_eval)
        self._setup_operating_mode_and_solver(experiment)

    def _determine_rebuild_requirement(self, build_on_eval: bool | None) -> bool:
        """Determine if model needs rebuilding on each evaluation."""
        if build_on_eval is not None:
            return build_on_eval
        if self.initial_state is not None:
            return True
        return self._check_geometric_parameters()

    def _setup_operating_mode_and_solver(self, experiment) -> None:
        """Setup operating mode and related configurations."""
        if experiment is not None:
            self._experiment = experiment
            self._operating_mode = OperatingMode.WITH_EXPERIMENT
        else:
            self._experiment = None
            self._operating_mode = OperatingMode.WITHOUT_EXPERIMENT
            self._model.events = []  # Turn off events

        if self._operating_mode == OperatingMode.WITH_EXPERIMENT:
            # Create the experiment simulation
            self._sim_experiment = self._create_experiment_simulation()

        elif self._solver.output_variables == []:
            # We can speed up the simulations using output_variables
            """DISABLE until PyBaMM PR 5118 is resolved"""
            # self._solver.output_variables = self._cost_names or []
            pass

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

    def _check_geometric_parameters(self) -> bool:
        """Check if parameters require model rebuilding."""
        if not self._pybop_parameters:
            return False

        model = self._model.new_copy()
        geometry = deepcopy(self._geometry)
        parameter_values = self._parameter_values.copy()

        # Set placeholder values for parameters
        for param in self._pybop_parameters:
            parameter_values.update({param.name: "[input]"})

        try:
            parameter_values.process_geometry(geometry=geometry)
            parameter_values.process_model(unprocessed_model=model)
            self._validate_geometric_parameters(geometry)
            self._parameter_values = parameter_values  # Update params w/ inputs
            return False
        except ValueError:
            return True

    def _validate_geometric_parameters(self, geometry) -> None:
        """
        Validate that geometry parameters are scalars.
        Credit: PyBaMM Team
        """
        for domain in geometry:
            for spatial_variable, spatial_limits in geometry[domain].items():
                # process tab information if using 1 or 2D current collectors
                if spatial_variable == "tabs":
                    for _, position_info in spatial_limits.items():
                        for _, sym in position_info.items():
                            self._validate_parameter_symbol(sym)
                else:
                    for _, sym in spatial_limits.items():
                        self._validate_parameter_symbol(sym)

    def _validate_parameter_symbol(self, sym) -> None:
        """
        Validate a single parameter symbol.
        Credit: PyBaMM Team
        """
        processed_sym = self._parameter_values.process_symbol(sym)
        leaves = processed_sym.post_order(filter=lambda node: len(node.children) == 0)

        for leaf in leaves:
            if not isinstance(leaf, pybamm.Scalar):
                raise ValueError("Geometry parameters must be Scalars")

    def rebuild(self, params: Inputs) -> None:
        """Build the PyBaMM pipeline using the given parameter_values."""
        # if there are no parameters to build, just return
        if self._built_model is not None and not self.requires_rebuild:
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
        """Build the PyBaMM pipeline using the given parameter_values."""
        model = self._model.new_copy()
        geometry = copy(self._geometry)

        if self.initial_state is not None:
            self._parameter_values.set_initial_state(
                self.initial_state, param=model.param, options=model.options
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

    def solve(
        self, calculate_sensitivities: bool = False
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
        inputs = self._pybop_parameters.to_pybamm_multiprocessing()

        if self._operating_mode == OperatingMode.WITHOUT_EXPERIMENT:
            solutions = self._solve_without_experiment(inputs, calculate_sensitivities)
        else:
            solutions = self._solve_with_experiment(inputs)

        return self._process_solutions(solutions)

    def _solve_without_experiment(
        self, inputs: list, calculate_sensitivities: bool
    ) -> list[pybamm.Solution]:
        """Solve without an experiment."""
        if self.requires_rebuild:
            return self._solve_with_rebuild(inputs)

        solutions = self._pybamm_solve(inputs, calculate_sensitivities)
        return solutions if isinstance(solutions, list) else [solutions]

    def _solve_with_rebuild(self, inputs: list) -> list[pybamm.Solution]:
        """Solve with model rebuilding for each parameter set."""
        solutions = []
        for params in inputs:
            self.rebuild(params)
            solution = self._pybamm_solve(params, False)
            solutions.append(solution)
        return solutions

    def _solve_with_experiment(self, inputs: list) -> list[pybamm.Solution]:
        """Solve with an experiment."""
        if self.requires_rebuild:
            return self._solve_experiment_with_rebuild(inputs)
        else:
            return self._solve_experiment_without_rebuild(inputs)

    def _solve_experiment_with_rebuild(self, inputs: list) -> list[pybamm.Solution]:
        """Solve by rebuilding simulation for each parameter set."""
        solutions = []

        for params in inputs:
            # Update parameters and create new simulation
            self._parameter_values.update(params)

            sim = self._create_experiment_simulation()
            solutions.append(sim.solve())

        return solutions

    def _solve_experiment_without_rebuild(self, inputs: list) -> list[pybamm.Solution]:
        """Solve using existing simulation with different input parameters."""
        solutions = []

        for params in inputs:
            solution = self._sim_experiment.solve(
                inputs=params, initial_soc=self.initial_state
            )
            solutions.append(solution)

        return solutions

    def _pybamm_solve(
        self, inputs: Inputs | list, calculate_sensitivities: bool
    ) -> list[pybamm.Solution]:
        """A function that runs the simulation using the built model."""

        return self._solver.solve(
            model=self._built_model,
            inputs=inputs,
            t_eval=self._t_eval,
            t_interp=self._t_interp,
            calculate_sensitivities=calculate_sensitivities,
        )

    def _process_solutions(
        self, solutions: list[pybamm.Solution]
    ) -> list[pybamm.Solution | FailedSolution]:
        """Convert failed solutions to FailedSolution objects."""
        processed_solutions = []
        for solution in solutions:
            if hasattr(solution, "termination") and solution.termination == "failure":
                failed_solution = FailedSolution(
                    self._cost_names, list(self._parameter_names)
                )
                processed_solutions.append(failed_solution)
            else:
                processed_solutions.append(solution)

        return processed_solutions

    @property
    def built_model(self):
        return self._built_model

    @property
    def parameter_names(self):
        return self._parameter_names

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
