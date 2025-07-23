import warnings
from copy import copy, deepcopy
from functools import lru_cache

import numpy as np
import pybamm

from pybop import Inputs, Parameters


class OperatingMode:
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

    To enable 2., you can pass a list of parameter names to the constructor, these parameters will be set
    before the model is built each time (using the `build` method).
    To enable 1, you can just pass an empty list. The model will be built once and subsequent calls
    to the `build` method will not change the model.
    """

    def __init__(
        self,
        model: pybamm.BaseModel,
        parameter_values: pybamm.ParameterValues | None = None,
        pybop_parameters: Parameters | None = None,
        solver: pybamm.BaseSolver | None = None,
        experiment: pybamm.Experiment = None,
        t_eval: np.ndarray = None,
        t_interp: np.ndarray | None = None,
        var_pts: dict | None = None,
        initial_state: float | str | None = None,
        build_on_eval: bool | None = None,
        save_at_cycles: list[int] = None,
    ):
        """
        Parameters
        ---------
        model : pybamm.BaseModel
            The PyBaMM model to be used.
        parameter_values : pybamm.ParameterValues
            The parameters to be used in the model.
        solver : pybamm.BaseSolver
            The solver to be used. If None, the idaklu solver will be used.
        t_eval : np.ndarray
            The time points to stop the solver at. These points should be used to inform the
            solver of discontinuities in the solution.
        t_interp : np.ndarray
            The time points at which to interpolate the solution. If None, no interpolation will be done.
        initial_state: float | str
            The initial state of charge or voltage for the battery model. If float, it will be represented
            as SoC and must be in range 0 to 1. If str, it will be represented as voltage and needs to be in
            the format: "3.4 V".
        build_on_eval : bool
            Boolean to determine if the model will be rebuilt every evaluation. If `initial_state` is provided,
            the model will be rebuilt every evaluation unless `build_on_eval` is `False`, in which case the model
            is built with the parameter values from construction only.
        """
        self._model = model
        self._model.events = []
        self._parameter_values = parameter_values or model.default_parameter_values
        self._pybop_parameters = pybop_parameters or Parameters([])
        self._parameter_names = self.pybop_parameters.keys()
        self._geometry = model.default_geometry
        self._var_pts = var_pts or model.default_var_pts
        self._spatial_methods = model.default_spatial_methods  # allow user input
        self._solver = pybamm.IDAKLUSolver() if solver is None else solver
        self._t_eval = t_eval
        self._t_interp = t_interp
        self._initial_state = initial_state
        self._built_initial_soc = None
        self._submesh_types = model.default_submesh_types  # allow user input
        self._built_model = self._model
        self.requires_rebuild = (
            build_on_eval
            if build_on_eval is not None
            else True
            if initial_state is not None
            else self._determine_rebuild()
        )
        self._setup_operating_mode(experiment)
        self._save_at_cycles = save_at_cycles
        self._callbacks = None
        self._starting_solution = None
        self._calc_esoh = False

        self.steps_to_built_models = None
        self.steps_to_built_solvers = None
        self.experiment_unique_steps_to_model = None
        self.get_esoh_solver = lru_cache()(self._get_esoh_solver)

    def _get_esoh_solver(self, calc_esoh):
        if calc_esoh is False:
            return None

        return pybamm.lithium_ion.ElectrodeSOHSolver(
            self._parameter_values, self._model.param, options=self._model.options
        )

    def _setup_operating_mode(self, experiment) -> None:
        if experiment is not None:
            self.operating_mode = OperatingMode.WITH_EXPERIMENT
            self.experiment = self._process_experiment(experiment)
        else:
            self.operating_mode = OperatingMode.WITHOUT_EXPERIMENT
            self._model.events = []  # Turn off events for non-experiment optimisation
            self.experiment = None

    def _process_experiment(self, experiment) -> pybamm.Experiment:
        if isinstance(experiment, str | pybamm.step.BaseStep):
            return pybamm.Experiment([experiment]).copy()
        elif isinstance(experiment, list):
            return pybamm.Experiment(experiment).copy()
        elif isinstance(experiment, pybamm.Experiment):
            return experiment.copy()
        else:
            raise TypeError("Invalid experiment type")

    def _determine_rebuild(self) -> bool:
        """
        The initial build process, useful to determine if
        rebuilding will be required.
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
        Checks whether the parameter values required a rebuild.
        This is reimplemented with only the required functionality.
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
        except ValueError:
            return True
        return False

    def _process_and_check(self, sym):
        """
        Process and check the geometry for each parameter.
        This is reimplemented as geometric parameters are not
        currently supported as InputParameters within Pybop.
        Credit: PyBaMM Team
        """
        new_sym = self._parameter_values.process_symbol(sym)
        leaves = new_sym.post_order(filter=lambda node: len(node.children) == 0)
        for leaf in leaves:
            if not isinstance(leaf, pybamm.Scalar):
                raise ValueError("Geometry parameters must be Scalars")
        return new_sym

    def rebuild(self, params: Inputs) -> None:
        """
        Build the PyBaMM pipeline using the given parameter_values.
        """
        # if there are no parameters to build, just return
        if not self.requires_rebuild:
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
        """
        Build the PyBaMM pipeline using the given parameter_values.
        """
        model = self._model.new_copy()

        if self._initial_state is not None:
            self._set_initial_state(model, self._initial_state)

        geometry = copy(self._geometry)
        if self.operating_mode is OperatingMode.WITH_EXPERIMENT:
            self.build_for_experiment(geometry)
        else:
            self._parameter_values.process_geometry(geometry)
            self._parameter_values.process_model(model)
            disc = self._create_mesh_and_discretisation(geometry)
            disc.process_model(model)
            self._built_model = model

        # reset the solver since we've built a new model
        self._solver = self._solver.copy()

    def _create_mesh_and_discretisation(self, geometry) -> pybamm.Discretisation:
        mesh = pybamm.Mesh(geometry, self._submesh_types, self._var_pts)
        disc = pybamm.Discretisation(mesh, self._spatial_methods, check_model=True)
        return disc

    def build_for_experiment(self, geometry) -> None:
        """
        Build the Pybamm pipeline for an experiment definition.
        """
        self._setup_experiment_models()
        self._parameter_values.process_geometry(geometry)
        disc = self._create_mesh_and_discretisation(geometry)
        self._build_experiment_steps(disc)

    def _setup_experiment_models(self) -> None:
        self._validate_experiment_parameters()
        parameter_values = self._parameter_values.copy()

        init_temp = self.experiment.steps[0].temperature
        if init_temp is not None:
            parameter_values["Initial temperature [K]"] = init_temp

        self.experiment_unique_steps_to_model = {}
        for step in self.experiment.unique_steps:
            parameterised_model = step.process_model(self._model, parameter_values)
            self.experiment_unique_steps_to_model[step.basic_repr()] = (
                parameterised_model
            )

        if self.experiment.initial_start_time:
            self._setup_rest_model(parameter_values)

    def _setup_rest_model(self, parameter_values) -> None:
        rest_step = pybamm.step.rest(duration=1)
        parameter_values["Ambient temperature [K]"] = "[input]"
        parameterised_model = rest_step.process_model(self._model, parameter_values)
        self.experiment_unique_steps_to_model["Rest for padding"] = parameterised_model

    def _build_experiment_steps(self, disc) -> None:
        self.steps_to_built_models = {}
        self.steps_to_built_solvers = {}

        for (
            step,
            model_with_set_params,
        ) in self.experiment_unique_steps_to_model.items():
            built_model = disc.process_model(model_with_set_params, inplace=True)
            solver = self._solver.copy()
            self.steps_to_built_models[step] = built_model
            self.steps_to_built_solvers[step] = solver

    def _validate_experiment_parameters(self) -> None:
        restrict_list = {"Initial temperature [K]", "Ambient temperature [K]"}

        for step in self.experiment.steps:
            if issubclass(step.__class__, pybamm.experiment.step.BaseStepImplicit):
                restrict_list.update(step.get_parameter_values([]).keys())
            elif issubclass(step.__class__, pybamm.experiment.step.BaseStepExplicit):
                restrict_list.update(["Current function [A]"])

        for key in restrict_list:
            if key in self._parameter_values.keys() and isinstance(
                self._parameter_values[key], pybamm.InputParameter
            ):
                raise pybamm.ModelError(
                    f"Cannot use '{key}' as input parameter. Restricted: {restrict_list}"
                )

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
        if self.operating_mode is OperatingMode.WITHOUT_EXPERIMENT:
            return self._solver.solve(
                model=self._built_model,
                inputs=self._pybop_parameters.to_dict(),
                t_eval=self._t_eval,
                t_interp=self._t_interp,
                calculate_sensitivities=calculate_sensitivities,
            )
        else:
            return self._solve_with_experiment(
                self._solver,
                self._calc_esoh,
                self._starting_solution,
                self._callbacks,
                self._pybop_parameters.to_dict(),
                # kwargs,
            )

    def _solve_with_experiment(
        self,
        solver,
        calc_esoh,
        starting_solution,
        callbacks,
        inputs,
        # kwargs,
    ):
        logs = {}
        kwargs = {}
        # callbacks.on_experiment_start(logs)

        # Initialise experiment execution
        experiment_runner = ExperimentRunner(
            self,
            solver,
            calc_esoh,
            callbacks,
            inputs,
            logs,
        )

        solution = experiment_runner.run(starting_solution, **kwargs)
        # callbacks.on_experiment_end(logs)
        return solution

    def run_padding_rest(self, kwargs, rest_time, step_solution, inputs):
        model = self.steps_to_built_models["Rest for padding"]
        solver = self.steps_to_built_solvers["Rest for padding"]

        # Make sure we take at least 2 timesteps. The period is hardcoded to 10
        # minutes,the user can always override it by adding a rest step
        npts = max(round(rest_time / 600) + 1, 2)

        step_solution_with_rest = solver.step(
            step_solution,
            model,
            rest_time,
            t_eval=np.linspace(0, rest_time, npts),
            save=False,
            inputs=inputs,
            **kwargs,
        )

        return step_solution_with_rest

    def _set_initial_state(self, model, initial_state) -> None:
        """
        Sets the initial state of the model.

        Parameters
        ----------
        model : pybamm.Model

        initial_state : float | str
            Can be either a float between 0 and 1 representing the initial SoC,
            or a string representing the initial voltage i.e. "3.4 V"
        """

        options = model.options
        param = model.param
        if options["open-circuit potential"] == "MSMR":
            self._parameter_values.set_initial_ocps(
                initial_state, param=param, options=options
            )
        elif options["working electrode"] == "positive":
            self._parameter_values.set_initial_stoichiometry_half_cell(
                initial_state,
                param=param,
                options=options,
                inputs=self._pybop_parameters.to_dict(),
            )
        else:
            self._parameter_values.set_initial_stoichiometries(
                initial_state,
                param=param,
                options=options,
                inputs=self._pybop_parameters.to_dict(),
            )

        # Save solved initial SOC in case we need to re-build the model
        self._built_initial_state = initial_state

    def set_parameter_value(self, key, value) -> None:
        self._parameter_values[key] = value

    @property
    def built_model(self):
        """The built Pybamm model."""
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

    @property
    def save_at_cycles(self):
        return self._save_at_cycles

    @property
    def t_interp(self):
        return self._t_interp


class ExperimentRunner:
    def __init__(self, simulation, solver, calc_esoh, callbacks, inputs, logs):
        self.sim: PybammPipeline = simulation
        self.solver: pybamm.BaseSolver = solver
        self.calc_esoh = calc_esoh
        self.save_at_cycles = self.sim.save_at_cycles
        self._callbacks = callbacks
        self.inputs = inputs
        self.logs = logs
        self.timer = pybamm.Timer()
        self.esoh_solver = simulation.get_esoh_solver(calc_esoh)

    def run(self, starting_solution, **kwargs):
        solution_data = self._initialise_solution_data(starting_solution)
        initial_start_time = self._setup_timing(starting_solution)

        return self._run_cycles(solution_data, initial_start_time, **kwargs)

    def _initialise_solution_data(self, starting_solution) -> dict:
        if starting_solution is None:
            return {"cycles": [], "summary_variables": [], "first_states": []}
        elif not hasattr(starting_solution, "all_summary_variables"):
            cycle_solution, cycle_sum_vars, cycle_first_state = (
                pybamm.make_cycle_solution(
                    [starting_solution],
                    esoh_solver=self.esoh_solver,
                    save_this_cycle=True,
                    inputs=self.inputs,
                )
            )
            return {
                "cycles": [cycle_solution],
                "summary_variables": [cycle_sum_vars],
                "first_states": [cycle_first_state],
            }
        else:
            return {
                "cycles": starting_solution.cycles.copy(),
                "summary_variables": starting_solution.all_summary_variables.copy(),
                "first_states": starting_solution.all_first_states.copy(),
            }

    def _setup_timing(self, starting_solution):
        if starting_solution is None:
            return self.sim.experiment.initial_start_time

        initial_start_time = starting_solution.initial_start_time
        if (
            initial_start_time is None
            and self.sim.experiment.initial_start_time is not None
        ):
            raise ValueError(
                "Starting solution needs start_time for experiments with start_time"
            )
        return initial_start_time

    def _run_cycles(self, solution_data, initial_start_time, **kwargs):
        cycle_lengths = self.sim.experiment.cycle_lengths
        solution = pybamm.EmptySolution()
        cycle_offset = len(solution_data["cycles"])
        stopping_conditions = self._setup_stopping_conditions()

        for cycle_num, cycle_length in enumerate(cycle_lengths, start=1):
            result = self._run_single_cycle(
                cycle_num,
                cycle_length,
                cycle_offset,
                solution_data,
                solution,
                initial_start_time,
                **kwargs,
            )

            if not result.get("continue", True):
                break

            solution = result.get("current_solution", solution)
            solution_data = result.get("current_solution_data", solution_data)

            if self._check_stopping_conditions(
                result.get("cycle_sum_vars", {}),
                stopping_conditions,
                result.get("min_voltage"),
            ):
                break

        return self._finalise_solution(solution, solution_data, initial_start_time)

    def _setup_stopping_conditions(self):
        termination = self.sim.experiment.termination
        conditions = {
            "voltage": termination.get("voltage"),
            "time": termination.get("time"),
            "capacity": termination.get("capacity"),
        }
        self.logs["stopping conditions"] = conditions
        return conditions

    def _run_single_cycle(
        self,
        cycle_num,
        cycle_length,
        cycle_offset,
        solution_data,
        current_solution,
        initial_start_time,
        **kwargs,
    ):
        cycle_solutions = []
        abs_cycle_num = cycle_offset + cycle_num
        save_this_cycle = (
            self.save_at_cycles is None or abs_cycle_num in self.save_at_cycles
        )

        # Initialise cycle state
        inputs = self.inputs.copy()
        min_voltage = None

        # Execute each step in the cycle
        for step_num, step in enumerate(
            self.sim.experiment.steps[
                sum(self.sim.experiment.cycle_lengths[: cycle_num - 1]) : sum(
                    self.sim.experiment.cycle_lengths[:cycle_num]
                )
            ]
        ):
            # Setup step inputs and timing
            step_inputs = self._setup_step_inputs(step, inputs, current_solution)

            # Handle rest padding if needed
            if (
                initial_start_time is not None
                and step_num == 0
                and abs_cycle_num == 1
                and current_solution.t[-1] < initial_start_time
            ):
                rest_time = initial_start_time - current_solution.t[-1]
                current_solution = self.sim.run_padding_rest(
                    kwargs, rest_time, current_solution, step_inputs
                )

            # Get model and solver for this step
            step_repr = step.basic_repr()
            model = self.sim.steps_to_built_models[step_repr]
            solver = self.sim.steps_to_built_solvers[step_repr]

            # Calculate step duration
            dt = step.duration or float("inf")

            step_t_eval, t_interp_processed = step.setup_timestepping(
                solver, dt, self.sim.t_interp
            )

            # Execute the step
            try:
                step_solution = solver.step(
                    current_solution,
                    model,
                    dt,
                    t_eval=step_t_eval,
                    t_interp=t_interp_processed,
                    save=save_this_cycle,
                    inputs=step_inputs,
                    **kwargs,
                )

                # Check for solver failure
                if step_solution.termination == "event" and hasattr(
                    step_solution, "termination_reason"
                ):
                    term_reason = step_solution.termination_reason
                    if "minimum voltage" in term_reason.lower():
                        min_voltage = np.min(step_solution["Terminal voltage [V]"].data)

                    # Check if this is a failure condition
                    if any(
                        fail_text in term_reason.lower()
                        for fail_text in ["failure", "error", "diverged"]
                    ):
                        self._callbacks.on_cycle_end(
                            self.logs,
                            abs_cycle_num,
                            cycle_summary={"termination": "solver_failure"},
                        )
                        return {
                            "continue": False,
                            "current_solution": current_solution,
                            "min_voltage": min_voltage,
                        }

                cycle_solutions.append(step_solution)
                current_solution = step_solution

            except Exception as e:
                pybamm.logger.error(
                    f"Step {step_num} in cycle {abs_cycle_num} failed: {e}"
                )
                # self._callbacks.on_cycle_end(
                #     self.logs,
                #     abs_cycle_num,
                #     cycle_summary={"termination": "step_failure", "error": str(e)},
                # )
                return {
                    "continue": False,
                    "current_solution": current_solution,
                    "min_voltage": min_voltage,
                }

            # Update min voltage tracking
            try:
                step_min_v = np.min(step_solution["Terminal voltage [V]"].data)
                min_voltage = (
                    step_min_v if min_voltage is None else min(min_voltage, step_min_v)
                )
            except KeyError:
                warnings.warn("Terminal voltage not found in solution", stacklevel=2)

        # Process cycle results
        if save_this_cycle and cycle_solutions:
            cycle_solution, cycle_sum_vars, cycle_first_state = (
                pybamm.make_cycle_solution(
                    cycle_solutions,
                    esoh_solver=self.esoh_solver,
                    save_this_cycle=True,
                    inputs=self.inputs,
                )
            )

            solution_data["cycles"].append(cycle_solution)
            solution_data["summary_variables"].append(cycle_sum_vars)
            solution_data["first_states"].append(cycle_first_state)

            # Log cycle completion
            # self._callbacks.on_cycle_end(self.logs, abs_cycle_num, cycle_sum_vars)

            return {
                "continue": True,
                "current_solution": current_solution,
                "cycle_sum_vars": cycle_sum_vars,
                "min_voltage": min_voltage,
                "solution_data": solution_data,
            }
        else:
            # For cycles not saved, still track basic info
            self._callbacks.on_cycle_end(self.logs, abs_cycle_num, {})

            return {
                "continue": True,
                "current_solution": current_solution,
                "min_voltage": min_voltage,
            }

    def _setup_step_inputs(self, step, base_inputs, current_solution):
        step_inputs = base_inputs.copy()

        # Set temperature input if specified
        if step.temperature is not None:
            step_inputs["Ambient temperature [K]"] = step.temperature

        # Handle start time for padding steps
        if (
            current_solution is not None
            and hasattr(current_solution, "t")
            and len(current_solution.t) > 0
        ):
            step_inputs["start_time"] = current_solution.t[-1]

        return step_inputs

    def _check_stopping_conditions(
        self, cycle_sum_vars, conditions, min_voltage
    ) -> bool:
        if conditions["capacity"] is not None:
            capacity = cycle_sum_vars.get("Capacity [A.h]")
            if capacity is not None and not np.isnan(capacity):
                if capacity <= conditions["capacity"]:
                    return True

        if conditions["voltage"] is not None and min_voltage is not None:
            if min_voltage <= conditions["voltage"][0]:
                return True

        return False

    def _finalise_solution(self, solution, solution_data, initial_start_time):
        if solution is not None and len(solution_data["cycles"]) > 0:
            solution.cycles = solution_data["cycles"]
            solution.update_summary_variables(solution_data["summary_variables"])
            solution.all_first_states = solution_data["first_states"]
            solution.initial_start_time = initial_start_time
        return solution
