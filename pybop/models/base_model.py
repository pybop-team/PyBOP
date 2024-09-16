import copy
from dataclasses import dataclass
from typing import Callable, Optional, Union

import casadi
import numpy as np
import pybamm
from pybamm import IDAKLUSolver as IDAKLUSolver
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from pybop import Dataset, Experiment, Parameters, ParameterSet, SymbolReplacer
from pybop.parameters.parameter import Inputs


@dataclass
class TimeSeriesState:
    """
    The current state of a time series model that is a pybamm model.
    """

    sol: pybamm.Solution
    inputs: Inputs
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


class BaseModel:
    """
    A base class for constructing and simulating models using PyBaMM.

    This class serves as a foundation for constructing models based on PyBaMM models. It
    provides methods to set up the model, define parameters, and perform simulations. The
    class is designed to be subclassed for creating models with custom behaviour.

    This class is based on PyBaMM's Simulation class. A PyBOP model is set up via a
    similar 3-step process. The `pybamm_model` attributes echoes the `model` attribute of
    a simulation, which tracks the model through the build process. Firstly, note that a
    PyBaMM `model` must first be built via `build_model` before a simulation or PyBOP
    model can be built. The 3-step process is then as follows.

    The `pybamm_model` attribute is first defined as an instance of the imported PyBaMM
    model, using any given model options. This initial version of the model is saved as
    the `_unprocessed_model` for future reference. Next, the type of each parameter in
    the parameter set as well as the geometry of the model is set. Parameters may be set
    as an input, interpolant, functional or just a standard PyBaMM parameter. This
    version of the model is referred to as the `model_with_set_params`. After its
    creation, the `pybamm_model` attribute is updated to point at this version of the
    model. Finally, the model required for simulations is built by defining the mesh and
    processing the discretisation. The complete model is referred to as the `built_model`
    and this version is used to run simulations.

    In order to rebuild a model with a different initial state or geometry, the
    `built_model` and the `model_with_set_params` must be cleared and the `pybamm_model`
    reset to the `unprocessed_model` in order to start the build process again.
    """

    def __init__(
        self,
        name: str = "Base Model",
        parameter_set: Optional[ParameterSet] = None,
        check_params: Callable = None,
        eis=False,
    ):
        """
        Initialise the BaseModel with an optional name and a parameter set.

        Parameters
        ----------
        name : str, optional
            The name given to the model instance.
        parameter_set : Union[pybop.ParameterSet, pybamm.ParameterValues], optional
            A dict-like object containing the parameter values.
        check_params : Callable, optional
            A compatibility check for the model parameters. Function, with
            signature
                check_params(
                    inputs: dict,
                    allow_infeasible_solutions: bool, optional
                )
            Returns true if parameters are valid, False otherwise. Can be
            used to impose constraints on valid parameters.

        Additional Attributes
        ---------------------
        pybamm_model : pybamm.BaseModel
            An instance of a PyBaMM model.
        parameters : pybop.Parameters
            The input parameters.
        param_check_counter : int
            A counter for the number of parameter checks (default: 0).
        allow_infeasible_solutions : bool, optional
            If True, parameter values will be simulated whether or not they are feasible
            (default: True).
        """
        self.name = name
        self.eis = eis
        if parameter_set is None:
            self._parameter_set = None
        elif isinstance(parameter_set, dict):
            self._parameter_set = pybamm.ParameterValues(parameter_set).copy()
        elif isinstance(parameter_set, pybamm.ParameterValues):
            self._parameter_set = parameter_set.copy()
        else:  # a pybop parameter set
            self._parameter_set = pybamm.ParameterValues(parameter_set.params).copy()
        self.param_checker = check_params

        self.pybamm_model = None
        self.parameters = Parameters()
        self.param_check_counter = 0
        self.allow_infeasible_solutions = True

    def build(
        self,
        parameters: Union[Parameters, dict] = None,
        inputs: Optional[Inputs] = None,
        initial_state: Optional[dict] = None,
        dataset: Optional[Dataset] = None,
        check_model: bool = True,
    ) -> None:
        """
        Construct the PyBaMM model, if not already built or if there are changes to any
        `rebuild_parameters` or the initial state.

        This method initializes the model components, applies the given parameters,
        sets up the mesh and discretisation if needed, and prepares the model
        for simulations.

        Parameters
        ----------
        parameters : pybop.Parameters or Dict, optional
            A pybop Parameters class or dictionary containing parameter values to apply to the model.
        inputs : Inputs
            The input parameters to be used when building the model.
        initial_state : dict, optional
            A valid initial state, e.g. the initial state of charge or open-circuit voltage.
            Defaults to None, indicating that the existing initial state of charge (for an ECM)
            or initial concentrations (for an EChem model) will be used.
            Accepted keys either `"Initial open-circuit voltage [V]"` or ``"Initial SoC"`
        dataset : pybop.Dataset or dict, optional
            The dataset to be used in the model construction.
        check_model : bool, optional
            If True, the model will be checked for correctness after construction.
        """
        if parameters is not None or inputs is not None:
            # Classify parameters and clear the model if rebuild required
            inputs = self.classify_parameters(parameters, inputs=inputs)

        if initial_state is not None:
            # Clear the model if rebuild required (currently if any initial state)
            self.set_initial_state(initial_state, inputs=inputs)

        if not self.pybamm_model._built:  # noqa: SLF001
            self.pybamm_model.build_model()

        if self.eis:
            self.set_up_for_eis(self.pybamm_model)
            self._parameter_set["Current function [A]"] = 0

            V_scale = getattr(self.pybamm_model.variables["Voltage [V]"], "scale", 1)
            I_scale = getattr(self.pybamm_model.variables["Current [A]"], "scale", 1)
            self.z_scale = self._parameter_set.evaluate(V_scale / I_scale)

        if dataset is not None and not self.eis:
            self.set_current_function(dataset)

        if self._built_model:
            return
        elif self.pybamm_model.is_discretised:
            self._model_with_set_params = self.pybamm_model
            self._built_model = self.pybamm_model
        else:
            self.set_parameters()
            self._mesh = pybamm.Mesh(self.geometry, self.submesh_types, self.var_pts)
            self._disc = pybamm.Discretisation(
                mesh=self.mesh,
                spatial_methods=self.spatial_methods,
                check_model=check_model,
            )
            self._built_model = self._disc.process_model(
                self._model_with_set_params, inplace=False
            )

            # Clear solver and setup model
            self._solver._model_set_up = {}  # noqa: SLF001

        self.n_states = self._built_model.len_rhs_and_alg  # len_rhs + len_alg

    def convert_to_pybamm_initial_state(self, initial_state: dict):
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
        if len(initial_state) > 1:
            raise ValueError("Expecting only one initial state.")
        elif "Initial SoC" in initial_state.keys():
            return initial_state["Initial SoC"]
        elif "Initial open-circuit voltage [V]" in initial_state.keys():
            return str(initial_state["Initial open-circuit voltage [V]"]) + "V"
        else:
            raise ValueError(f'Unrecognised initial state: "{list(initial_state)[0]}"')

    def set_initial_state(self, initial_state: dict, inputs: Optional[Inputs] = None):
        """
        Set the initial state of charge or concentrations for the battery model.

        Parameters
        ----------
        initial_state : dict
            A valid initial state, e.g. the initial state of charge or open-circuit voltage.
        inputs : Inputs
            The input parameters to be used when building the model.
        """
        self.clear()

        initial_state = self.convert_to_pybamm_initial_state(initial_state)

        if isinstance(self.pybamm_model, pybamm.equivalent_circuit.Thevenin):
            initial_state = self.get_initial_state(initial_state, inputs=inputs)
            self._unprocessed_parameter_set.update({"Initial SoC": initial_state})

        else:
            if not self.pybamm_model._built:  # noqa: SLF001
                self.pybamm_model.build_model()

            # Temporary construction of attributes for PyBaMM
            self.model = self._model = self.pybamm_model
            self._unprocessed_parameter_values = self._unprocessed_parameter_set

            # Set initial state via PyBaMM's Simulation class
            pybamm.Simulation.set_initial_soc(self, initial_state, inputs=inputs)

            # Update the default parameter set for consistency
            self._unprocessed_parameter_set = self._parameter_values

            # Clear the pybamm objects
            del self.model  # can be removed after PyBaMM's next release, fixed with pybamm-team/PyBaMM#4319
            del self._model
            del self._unprocessed_parameter_values
            del self._parameter_values

        # Use a copy of the updated default parameter set
        self._parameter_set = self._unprocessed_parameter_set.copy()

    def set_current_function(self, dataset: Union[Dataset, dict]):
        """
        Update the input current function according to the data.

        Parameters
        ----------
        dataset : pybop.Dataset or dict, optional
            The dataset to be used in the model construction.
        """
        if "Current function [A]" in self._parameter_set.keys():
            if "Current function [A]" not in self.parameters.keys():
                current = pybamm.Interpolant(
                    dataset["Time [s]"],
                    dataset["Current function [A]"],
                    pybamm.t,
                )
                # Update both the active and unprocessed parameter sets for consistency
                self._parameter_set["Current function [A]"] = current
                self._unprocessed_parameter_set["Current function [A]"] = current

    def set_parameters(self):
        """
        Assign the parameters to the model.

        This method processes the model with the given parameters, sets up
        the geometry, and updates the model instance.
        """
        if self._model_with_set_params:
            return

        self._model_with_set_params = self._parameter_set.process_model(
            self._unprocessed_model, inplace=False
        )
        self._parameter_set.process_geometry(self._geometry)
        self.pybamm_model = self._model_with_set_params

    def set_up_for_eis(self, model):
        """
        Set up the model for electrochemical impedance spectroscopy (EIS) simulations.

        This method sets up the model for EIS simulations by adding the necessary
        algebraic equations and variables to the model.
        Originally developed by pybamm-eis: https://github.com/pybamm-team/pybamm-eis

        Parameters
        ----------
        model : pybamm.Model
            The PyBaMM model to be used for EIS simulations.
        """
        V_cell = pybamm.Variable("Voltage variable [V]")
        model.variables["Voltage variable [V]"] = V_cell
        V = model.variables["Voltage [V]"]

        # Add algebraic equation for the voltage
        model.algebraic[V_cell] = V_cell - V
        model.initial_conditions[V_cell] = model.param.ocv_init

        # Create the FunctionControl submodel and extract variables
        external_circuit_variables = pybamm.external_circuit.FunctionControl(
            model.param, None, model.options, control="algebraic"
        ).get_fundamental_variables()

        # Perform the replacement
        symbol_replacement_map = {
            model.variables[name]: variable
            for name, variable in external_circuit_variables.items()
        }

        # Don't replace initial conditions, as these should not contain
        # Variable objects
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

    def clear(self):
        """
        Clear any built PyBaMM model.
        """
        self._model_with_set_params = None
        self._built_model = None
        self._built_initial_soc = None
        self._mesh = None
        self._disc = None

    def classify_parameters(
        self, parameters: Optional[Parameters] = None, inputs: Optional[Inputs] = None
    ):
        """
        Check for any 'rebuild_parameters' which require a model rebuild and
        update the unprocessed_parameter_set if a rebuild is required.

        Parameters
        ----------
        parameters : Parameters, optional
            The optimisation parameters. Defaults to None, resulting in the internal
            `pybop.Parameters` object to be used.
        inputs : Inputs, optional
            The input parameters for the simulation (default: None).
        """
        self.parameters = parameters or self.parameters

        # Compile all parameters and inputs
        parameter_dictionary = self.parameters.as_dict()
        parameter_dictionary.update(inputs or {})

        rebuild_parameters = {
            param: parameter_dictionary[param]
            for param in parameter_dictionary
            if param in self.geometric_parameters
        }
        standard_parameters = {
            param: parameter_dictionary[param]
            for param in parameter_dictionary
            if param not in self.geometric_parameters
        }

        # Mark any standard parameters in the active parameter set and pass as inputs
        for key in standard_parameters.keys():
            self._parameter_set[key] = "[input]"

        # Clear any built model, update the parameter set and geometry if rebuild required
        if rebuild_parameters:
            requires_rebuild = False
            # A rebuild is required if any of the rebuild parameter values have changed
            for key, value in rebuild_parameters.items():
                if value != self._unprocessed_parameter_set[key]:
                    requires_rebuild = True
            if requires_rebuild:
                self.clear()
                self._geometry = self.pybamm_model.default_geometry
                # Update both the active and unprocessed parameter sets for consistency
                self._parameter_set.update(rebuild_parameters)
                self._unprocessed_parameter_set.update(rebuild_parameters)

        return standard_parameters

    def reinit(
        self, inputs: Inputs, t: float = 0.0, x: Optional[np.ndarray] = None
    ) -> TimeSeriesState:
        """
        Initialises the solver with the given inputs and returns the initial state of the problem
        """
        if self._built_model is None:
            raise ValueError("Model must be built before calling reinit")

        inputs = self.parameters.verify(inputs)

        self._solver.set_up(self._built_model, inputs=inputs)

        if x is None:
            x = self._built_model.y0

        return self.get_state(inputs, t, x)

    def get_state(self, inputs: Inputs, t: float, x: np.ndarray) -> TimeSeriesState:
        """
        Returns the given state for the problem (inputs are assumed constant since last reinit)
        """
        if self._built_model is None:
            raise ValueError("Model must be built before calling get_state")

        sol = pybamm.Solution([np.asarray([t])], [x], self._built_model, inputs)

        return TimeSeriesState(sol=sol, inputs=inputs, t=t)

    def step(self, state: TimeSeriesState, time: np.ndarray) -> TimeSeriesState:
        """
        Step forward in time from the given state until the given time.

        Parameters
        ----------
        state : TimeSeriesState
            The current state of the model
        time : np.ndarray
            The time to simulate the system until (in whatever time units the model is in)
        """
        dt = time - state.t
        new_sol = self._solver.step(
            state.sol, self._built_model, dt, npts=2, inputs=state.inputs, save=False
        )
        return TimeSeriesState(sol=new_sol, inputs=state.inputs, t=time)

    def simulate(
        self, inputs: Inputs, t_eval: np.array, initial_state: Optional[dict] = None
    ) -> Union[pybamm.Solution, list[np.float64]]:
        """
        Execute the forward model simulation and return the result.

        Parameters
        ----------
        inputs : Inputs
            The input parameters for the simulation.
        t_eval : array-like
            An array of time points at which to evaluate the solution.
        initial_state : dict, optional
            A valid initial state, e.g. the initial state of charge or open-circuit voltage.
            Defaults to None, indicating that the existing initial state of charge (for an ECM)
            or initial concentrations (for an EChem model) will be used.

        Returns
        -------
        pybamm.Solution
            The solution object returned by a PyBaMM simulation, or a pybamm error in the case
            where the parameter values are infeasible and infeasible solutions are not allowed.

        Raises
        ------
        ValueError
            If the model has not been built before simulation.
        """
        inputs = self.parameters.verify(inputs)

        # Build or rebuild if required
        self.build(inputs=inputs, initial_state=initial_state)

        if not self.check_params(
            inputs=inputs,
            allow_infeasible_solutions=self.allow_infeasible_solutions,
        ):
            raise ValueError("These parameter values are infeasible.")

        return self.solver.solve(
            self._built_model,
            inputs=inputs,
            t_eval=[t_eval[0], t_eval[-1]]
            if isinstance(self._solver, IDAKLUSolver)
            else t_eval,
            t_interp=t_eval,
        )

    def simulateEIS(
        self, inputs: Inputs, f_eval: list, initial_state: Optional[dict] = None
    ) -> dict[str, np.ndarray]:
        """
        Compute the forward model simulation with electrochemical impedance spectroscopy
        and return the result.

        Parameters
        ----------
        inputs : dict or array-like
            The input parameters for the simulation. If array-like, it will be
            converted to a dictionary using the model's fit keys.
        f_eval : array-like
            An array of frequency points at which to evaluate the solution.

        Returns
        -------
        array-like
            The simulation result corresponding to the specified signal.

        Raises
        ------
        ValueError
            If the model has not been built before simulation.
        """
        inputs = self.parameters.verify(inputs)

        # Build or rebuild if required
        self.build(inputs=inputs, initial_state=initial_state)

        if not self.check_params(
            inputs=inputs,
            allow_infeasible_solutions=self.allow_infeasible_solutions,
        ):
            raise ValueError("These parameter values are infeasible.")

        self.initialise_eis_simulation(inputs)
        zs = [self.calculate_impedance(frequency) for frequency in f_eval]

        return {"Impedance": np.asarray(zs) * self.z_scale}

    def initialise_eis_simulation(self, inputs: Optional[Inputs] = None):
        """
        Initialise the Electrochemical Impedance Spectroscopy (EIS) simulation.

        This method sets up the mass matrix and solver, converts inputs to the appropriate format,
        extracts necessary attributes from the model, and prepares matrices for the simulation.

        Parameters
        ----------
        inputs : dict (optional)
            The input parameters for the simulation.
        """
        # Setup mass matrix, solver
        self.M = self._built_model.mass_matrix.entries
        self._solver.set_up(self._built_model, inputs=inputs)

        # Convert inputs to casadi format if needed
        casadi_inputs = (
            casadi.vertcat(*inputs.values())
            if inputs is not None and self._built_model.convert_to_format == "casadi"
            else inputs or []
        )

        # Extract necessary attributes from the model
        self.y0 = self._built_model.concatenated_initial_conditions.evaluate(
            0, inputs=inputs
        )
        self.J = self._built_model.jac_rhs_algebraic_eval(
            0, self.y0, casadi_inputs
        ).sparse()

        # Convert to Compressed Sparse Column format
        self.M = csc_matrix(self.M)
        self.J = csc_matrix(self.J)

        # Add forcing to the RHS on the current density
        self.b = np.zeros(self.y0.shape)
        self.b[-1] = -1

    def calculate_impedance(self, frequency):
        """
        Calculate the impedance for a given frequency.

        This method computes the system matrix, solves the linear system, and calculates
        the impedance based on the solution.

        Parameters
        ----------
            frequency (np.ndarray | list like): The frequency at which to calculate the impedance.

        Returns
        -------
            The calculated impedance (complex np.ndarray).
        """
        # Compute the system matrix
        A = 1.0j * 2 * np.pi * frequency * self.M - self.J

        # Solve the system
        x = spsolve(A, self.b)

        # Calculate the impedance
        return -x[-2] / x[-1]

    def simulateS1(
        self, inputs: Inputs, t_eval: np.array, initial_state: Optional[dict] = None
    ):
        """
        Perform the forward model simulation with sensitivities.

        Parameters
        ----------
        inputs : Inputs
            The input parameters for the simulation.
        t_eval : array-like
            An array of time points at which to evaluate the solution and its
            sensitivities.
        initial_state : dict, optional
            A valid initial state, e.g. the initial state of charge or open-circuit voltage.
            Defaults to None, indicating that the existing initial state of charge (for an ECM)
            or initial concentrations (for an EChem model) will be used.

        Returns
        -------
        pybamm.Solution
            The solution object returned by a PyBaMM simulation, or a pybamm error in the case
            where the parameter values are infeasible and infeasible solutions are not allowed.

        Raises
        ------
        ValueError
            If the model has not been built before simulation.
        """
        inputs = self.parameters.verify(inputs)

        if initial_state is not None or any(
            key in self.geometric_parameters for key in inputs.keys()
        ):
            raise ValueError(
                "Cannot use sensitivities for parameters which require a model rebuild"
            )

        # Build if required
        self.build(inputs=inputs, initial_state=initial_state)

        if not self.check_params(
            inputs=inputs,
            allow_infeasible_solutions=self.allow_infeasible_solutions,
        ):
            raise ValueError("These parameter values are infeasible.")

        return self._solver.solve(
            self._built_model,
            inputs=inputs,
            t_eval=[t_eval[0], t_eval[-1]]
            if isinstance(self._solver, IDAKLUSolver)
            else t_eval,
            calculate_sensitivities=True,
            t_interp=t_eval,
        )

    def predict(
        self,
        inputs: Optional[Inputs] = None,
        t_eval: Optional[np.array] = None,
        parameter_set: Optional[ParameterSet] = None,
        experiment: Optional[Experiment] = None,
        initial_state: Optional[dict] = None,
    ) -> dict[str, np.ndarray[np.float64]]:
        """
        Solve the model using PyBaMM's simulation framework and return the solution.

        This method sets up a PyBaMM simulation by configuring the model, parameters, experiment
        or time vector, and initial state of charge (if provided). Either 't_eval' or 'experiment'
        must be provided. It then solves the simulation and returns the resulting solution object.

        Parameters
        ----------
        inputs : Inputs, optional
            Input parameters for the simulation. Defaults to None, indicating that the
            default parameters should be used.
        t_eval : array-like, optional
            An array of time points at which to evaluate the solution. Defaults to None,
            which means the time points need to be specified within experiment or elsewhere.
        parameter_set : Union[pybop.ParameterSet, pybamm.ParameterValues], optional
            A dict-like object containing the parameter values to use for the simulation.
            Defaults to the model's current ParameterValues if None.
        experiment : pybamm.Experiment, optional
            A PyBaMM Experiment object specifying the experimental conditions under which
            the simulation should be run. Defaults to None, indicating no experiment.
        initial_state : dict, optional
            A valid initial state, e.g. the initial state of charge or open-circuit voltage.
            Defaults to None, indicating that the existing initial state of charge (for an ECM)
            or initial concentrations (for an EChem model) will be used.

        Returns
        -------
        pybamm.Solution
            The solution object returned by a PyBaMM simulation, or a pybamm error in the case
            where the parameter values are infeasible and infeasible solutions are not allowed.

        Raises
        ------
        ValueError
            If the model has not been configured properly before calling this method or
            if PyBaMM models are not supported by the current simulation method.

        """
        if self.pybamm_model is None:
            raise ValueError(
                "The predict method currently only supports PyBaMM models."
            )
        elif not self._unprocessed_model._built:  # noqa: SLF001
            self._unprocessed_model.build_model()

        no_parameter_set = parameter_set is None
        parameter_set = parameter_set or self._unprocessed_parameter_set.copy()
        if inputs is not None:
            inputs = self.parameters.verify(inputs)
            parameter_set.update(inputs)

        if initial_state is not None:
            if no_parameter_set:
                # Update the default initial state for consistency
                self.set_initial_state(initial_state)

            initial_state = self.convert_to_pybamm_initial_state(initial_state)
            if isinstance(self.pybamm_model, pybamm.equivalent_circuit.Thevenin):
                parameter_set["Initial SoC"] = self._parameter_set["Initial SoC"]
                initial_state = None

        if not self.check_params(
            parameter_set=parameter_set,
            allow_infeasible_solutions=self.allow_infeasible_solutions,
        ):
            raise ValueError("These parameter values are infeasible.")

        if experiment is not None:
            return pybamm.Simulation(
                model=self._unprocessed_model,
                experiment=experiment,
                parameter_values=parameter_set,
            ).solve(initial_soc=initial_state)
        elif t_eval is not None:
            return pybamm.Simulation(
                model=self._unprocessed_model,
                parameter_values=parameter_set,
            ).solve(t_eval=t_eval, initial_soc=initial_state)
        else:
            raise ValueError(
                "The predict method requires either an experiment or t_eval "
                "to be specified."
            )

    def check_params(
        self,
        inputs: Optional[Inputs] = None,
        parameter_set: Optional[ParameterSet] = None,
        allow_infeasible_solutions: bool = True,
    ):
        """
        Check compatibility of the model parameters.

        Parameters
        ----------
        inputs : Inputs
            The input parameters for the simulation.
        parameter_set : Union[pybop.ParameterSet, pybamm.ParameterValues], optional
            A dict-like object containing the parameter values.
        allow_infeasible_solutions : bool, optional
            If True, infeasible parameter values will be allowed in the optimisation (default: True).

        Returns
        -------
        bool
            A boolean which signifies whether the parameters are compatible.

        """
        inputs = self.parameters.verify(inputs) or {}
        parameter_set = parameter_set or self._parameter_set

        return self._check_params(
            inputs=inputs,
            parameter_set=parameter_set,
            allow_infeasible_solutions=allow_infeasible_solutions,
        )

    def _check_params(
        self,
        inputs: Inputs,
        parameter_set: ParameterSet,
        allow_infeasible_solutions: bool = True,
    ):
        """
        A compatibility check for the model parameters which can be implemented by subclasses
        if required, otherwise it returns True by default.

        Parameters
        ----------
        inputs : Inputs
            The input parameters for the simulation.
        parameter_set : Union[pybop.ParameterSet, pybamm.ParameterValues], optional
            A dict-like object containing the parameter values.
        allow_infeasible_solutions : bool, optional
            If True, infeasible parameter values will be allowed in the optimisation (default: True).

        Returns
        -------
        bool
            A boolean which signifies whether the parameters are compatible.
        """
        if self.param_checker:
            return self.param_checker(inputs, allow_infeasible_solutions)
        return True

    def copy(self):
        """
        Return a copy of the model.

        Returns
        -------
        BaseModel
            A copy of the model.
        """
        return copy.copy(self)

    def new_copy(self):
        """
        Return a new copy of the model, explicitly copying all the mutable attributes
        to avoid issues with shared objects.

        Returns
        -------
        BaseModel
            A new copy of the model.
        """
        model_class = type(self)
        if self.pybamm_model is None:
            model_args = {"parameter_set": self._parameter_set.copy()}
        else:
            model_args = {
                "options": self._unprocessed_model.options,
                "parameter_set": self._unprocessed_parameter_set.copy(),
                "geometry": self.pybamm_model.default_geometry.copy(),
                "submesh_types": self.pybamm_model.default_submesh_types.copy(),
                "var_pts": self.pybamm_model.default_var_pts.copy(),
                "spatial_methods": self.pybamm_model.default_spatial_methods.copy(),
                "solver": self.pybamm_model.default_solver.copy(),
                "eis": copy.copy(self.eis),
            }

        return model_class(**model_args)

    def get_parameter_info(self, print_info: bool = False):
        """
        Extracts the parameter names and types and returns them as a dictionary.
        """
        if not self.pybamm_model._built:  # noqa: SLF001
            self.pybamm_model.build_model()

        info = self.pybamm_model.get_parameter_info()

        reduced_info = dict()
        for param, param_type in info.values():
            param_name = getattr(param, "name", str(param))
            reduced_info[param_name] = param_type

        if print_info:
            for param, param_type in info.values():
                print(param, " : ", param_type)

        return reduced_info

    def cell_mass(self, parameter_set: ParameterSet = None):
        """
        Calculate the cell mass in kilograms.

        This method must be implemented by subclasses.

        Parameters
        ----------
        parameter_set : Union[pybop.ParameterSet, pybamm.ParameterValues], optional
            A dict-like object containing the parameter values.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the subclass.
        """
        raise NotImplementedError

    def cell_volume(self, parameter_set: ParameterSet = None):
        """
        Calculate the cell volume in m3.

        This method must be implemented by subclasses.

        Parameters
        ----------
        parameter_set : Union[pybop.ParameterSet, pybamm.ParameterValues], optional
            A dict-like object containing the parameter values.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the subclass.
        """
        raise NotImplementedError

    def approximate_capacity(self, parameter_set: ParameterSet = None):
        """
        Calculate a new estimate for the nominal capacity based on the theoretical energy
        density and an average voltage.

        This method must be implemented by subclasses.

        Parameters
        ----------
        parameter_set : Union[pybop.ParameterSet, pybamm.ParameterValues], optional
            A dict-like object containing the parameter values.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the subclass.
        """
        raise NotImplementedError

    @property
    def built_model(self):
        return self._built_model

    @property
    def built_initial_soc(self):
        return self._built_initial_soc

    @property
    def parameter_set(self):
        return self._parameter_set

    @property
    def model_with_set_params(self):
        return self._model_with_set_params

    @property
    def geometry(self):
        return self._geometry

    @property
    def submesh_types(self):
        return self._submesh_types

    @property
    def mesh(self):
        return self._mesh

    @property
    def disc(self):
        return self._disc

    @property
    def var_pts(self):
        return self._var_pts

    @property
    def spatial_methods(self):
        return self._spatial_methods

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, solver):
        self._solver = solver.copy() if solver is not None else None
