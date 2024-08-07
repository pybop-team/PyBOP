import copy
from dataclasses import dataclass
from typing import Optional, Union

import casadi
import numpy as np
import pybamm

from pybop import Dataset, Experiment, Parameters, ParameterSet
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
        self, name: str = "Base Model", parameter_set: Optional[ParameterSet] = None
    ):
        """
        Initialise the BaseModel with an optional name and a parameter set.

        Parameters
        ----------
        name : str, optional
            The name given to the model instance.
        parameter_set : pybop.ParameterSet, optional
            A PyBOP ParameterSet, PyBaMM ParameterValues object or a dictionary containing the
            parameter values.

        Additional Attributes
        ---------------------
        parameters : pybop.Parameters
            The input parameters.
        output_variables : list[str], optional
            A list of names of variables to include in the solution object.
        rebuild_parameters : dict
            A list of parameters which require the model to be rebuilt (default: {}).
        standard_parameters : dict
            A list of standard (i.e. not rebuild) parameters (default: {}).
        param_check_counter : int
            A counter for the number of parameter checks (default: 0).
        allow_infeasible_solutions : bool, optional
            If True, parameter values will be simulated whether or not they are feasible
            (default: True).
        """
        self.name = name
        if parameter_set is None:
            self._parameter_set = None
        elif isinstance(parameter_set, dict):
            self._parameter_set = pybamm.ParameterValues(parameter_set).copy()
        elif isinstance(parameter_set, pybamm.ParameterValues):
            self._parameter_set = parameter_set.copy()
        else:  # a pybop parameter set
            self._parameter_set = pybamm.ParameterValues(parameter_set.params).copy()

        self.pybamm_model = None
        self.parameters = Parameters()
        self.rebuild_parameters = {}
        self.standard_parameters = {}
        self.param_check_counter = 0
        self.allow_infeasible_solutions = True
        self.current_function = None

    def build(
        self,
        dataset: Optional[Dataset] = None,
        parameters: Union[Parameters, dict] = None,
        check_model: bool = True,
        initial_state: Optional[dict] = None,
    ) -> None:
        """
        Construct the PyBaMM model if not already built, and set parameters.

        This method initializes the model components, applies the given parameters,
        sets up the mesh and discretisation if needed, and prepares the model
        for simulations.

        Parameters
        ----------
        dataset : pybamm.Dataset, optional
            The dataset to be used in the model construction.
        parameters : pybop.Parameters or Dict, optional
            A pybop Parameters class or dictionary containing parameter values to apply to the model.
        check_model : bool, optional
            If True, the model will be checked for correctness after construction.
        initial_state : dict, optional
            A valid initial state, e.g. the initial state of charge or open-circuit voltage.
            Defaults to None, indicating that the existing initial state of charge (for an ECM)
            or initial concentrations (for an EChem model) will be used.
        """
        if parameters is not None:
            self.parameters = parameters
            self.classify_and_update_parameters(self.parameters)

        if initial_state is not None:
            self.set_initial_state(initial_state)

        if self._built_model:
            return

        elif self.pybamm_model.is_discretised:
            self._model_with_set_params = self.pybamm_model
            self._built_model = self.pybamm_model

        else:
            if not self.pybamm_model._built:  # noqa: SLF001
                self.pybamm_model.build_model()
            self.set_params(dataset=dataset)

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

    def set_initial_state(self, initial_state: dict):
        """
        Set the initial state of charge or concentrations for the battery model.

        Parameters
        ----------
        initial_state : dict
            A valid initial state, e.g. the initial state of charge or open-circuit voltage.
        """
        initial_state = self.convert_to_pybamm_initial_state(initial_state)

        if isinstance(self.pybamm_model, pybamm.equivalent_circuit.Thevenin):
            initial_state = self.get_initial_state(initial_state)
            self._unprocessed_parameter_set.update({"Initial SoC": initial_state})

        else:
            # Temporary construction of attribute for PyBaMM
            self.model = self._model = self.pybamm_model
            self._unprocessed_parameter_values = self._unprocessed_parameter_set

            # Set initial SOC via PyBaMM's Simulation class
            pybamm.Simulation.set_initial_soc(self, initial_state, inputs=None)

            # Update the default parameter set for consistency
            self._unprocessed_parameter_set = self._parameter_values

            # Clear the pybamm objects
            del self.model  # can be removed after PyBaMM's next release, fixed with pybamm-team/PyBaMM#4319
            del self._model
            del self._unprocessed_parameter_values
            del self._parameter_values

        # Use a copy of the updated default parameter set
        self._parameter_set = self._unprocessed_parameter_set.copy()

    def set_params(self, rebuild: bool = False, dataset: Dataset = None):
        """
        Assign the parameters to the model.

        This method processes the model with the given parameters, sets up
        the geometry, and updates the model instance.
        """
        if self.model_with_set_params and not rebuild:
            return

        # Mark any simulation inputs in the parameter set
        for key in self.standard_parameters.keys():
            self._parameter_set[key] = "[input]"

        if "Current function [A]" in self._parameter_set.keys():
            if dataset is not None and (not self.rebuild_parameters or not rebuild):
                if "Current function [A]" not in self.parameters.keys():
                    self.current_function = pybamm.Interpolant(
                        dataset["Time [s]"],
                        dataset["Current function [A]"],
                        pybamm.t,
                    )
                    self._parameter_set["Current function [A]"] = self.current_function
            elif rebuild and self.current_function is not None:
                self._parameter_set["Current function [A]"] = self.current_function

        self._model_with_set_params = self._parameter_set.process_model(
            self._unprocessed_model, inplace=False
        )
        if self.geometry is not None:
            self._parameter_set.process_geometry(self.geometry)
        self.pybamm_model = self._model_with_set_params

    def clear(self):
        """
        Clear any built PyBaMM model.
        """
        self._model_with_set_params = None
        self._built_model = None
        self._built_initial_soc = None
        self._mesh = None
        self._disc = None

    def rebuild(
        self,
        dataset: Optional[Dataset] = None,
        parameters: Union[Parameters, dict] = None,
        check_model: bool = True,
        initial_state: Optional[dict] = None,
    ) -> None:
        """
        Rebuild the PyBaMM model for a given set of inputs.

        This method requires the self.build() method to be called first, and
        then rebuilds the model for a given parameter set. Specifically,
        this method applies the given parameters, sets up the mesh and
        discretisation if needed, and prepares the model for simulations.

        Parameters
        ----------
        dataset : pybamm.Dataset, optional
            The dataset to be used in the model construction.
        parameters : pybop.Parameters or Dict, optional
            A pybop Parameters class or dictionary containing parameter values to apply to the model.
        check_model : bool, optional
            If True, the model will be checked for correctness after construction.
        initial_state : dict, optional
            A valid initial state, e.g. the initial state of charge or open-circuit voltage.
            Defaults to None, indicating that the existing initial state of charge (for an ECM)
            or initial concentrations (for an EChem model) will be used.
        """
        if parameters is not None:
            self.classify_and_update_parameters(parameters)

        if initial_state is not None:
            self.set_initial_state(initial_state)

        if self._built_model is None:
            raise ValueError("Model must be built before calling rebuild")

        self.set_params(rebuild=True, dataset=dataset)
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

    def classify_and_update_parameters(self, parameters: Parameters):
        """
        Update the parameter values according to their classification as either
        'rebuild_parameters' which require a model rebuild and
        'standard_parameters' which do not.

        Parameters
        ----------
        parameters : pybop.Parameters
            The input parameters.
        """
        self.parameters = parameters or Parameters()

        parameter_dictionary = self.parameters.as_dict()

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

        self.rebuild_parameters.update(rebuild_parameters)
        self.standard_parameters.update(standard_parameters)

        if self.rebuild_parameters:
            self._geometry = self.pybamm_model.default_geometry

        # Update both the active and unprocessed parameter sets for consistency
        if self._parameter_set is not None:
            self._parameter_set.update(parameter_dictionary)
            self._unprocessed_parameter_set = self._parameter_set

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

        if self._built_model is None:
            raise ValueError("Model must be built before calling simulate")

        requires_rebuild = False
        # A rebuild is required if any of the rebuild parameter values have changed
        for key, value in inputs.items():
            if key in self.rebuild_parameters:
                if value != self.parameters[key].value:
                    requires_rebuild = True
        # Or if the simulation is set to start from a specific initial value
        if initial_state is not None:
            requires_rebuild = True

        if requires_rebuild:
            self.parameters.update(values=list(inputs.values()))
            self.rebuild(parameters=self.parameters, initial_state=initial_state)

        if not self.check_params(
            inputs=inputs,
            allow_infeasible_solutions=self.allow_infeasible_solutions,
        ):
            raise ValueError("These parameter values are infeasible.")

        return self.solver.solve(self._built_model, inputs=inputs, t_eval=t_eval)

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

        if self._built_model is None:
            raise ValueError("Model must be built before calling simulate")

        if self.rebuild_parameters or initial_state is not None:
            raise ValueError(
                "Cannot use sensitivies for parameters which require a model rebuild"
            )

        if not self.check_params(
            inputs=inputs,
            allow_infeasible_solutions=self.allow_infeasible_solutions,
        ):
            raise ValueError("These parameter values are infeasible.")

        return self._solver.solve(
            self._built_model,
            inputs=inputs,
            t_eval=t_eval,
            calculate_sensitivities=True,
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
        parameter_set : pybamm.ParameterValues, optional
            A PyBaMM ParameterValues object or a dictionary containing the parameter values
            to use for the simulation. Defaults to the model's current ParameterValues if None.
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
        if self._unprocessed_model is None:
            raise ValueError(
                "The predict method currently only supports PyBaMM models."
            )
        elif not self._unprocessed_model._built:  # noqa: SLF001
            self._unprocessed_model.build_model()

        parameter_set = parameter_set or self._unprocessed_parameter_set.copy()
        if inputs is not None:
            inputs = self.parameters.verify(inputs)
            parameter_set.update(inputs)

        if initial_state is not None:
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
        parameter_set : pybop.ParameterSet, optional
            A PyBOP parameter set object or a dictionary containing the parameter values.
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
        parameter_set : pybop.ParameterSet
            A PyBOP parameter set object or a dictionary containing the parameter values.
        allow_infeasible_solutions : bool, optional
            If True, infeasible parameter values will be allowed in the optimisation (default: True).

        Returns
        -------
        bool
            A boolean which signifies whether the parameters are compatible.
        """
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
        parameter_set : dict, optional
            A dictionary containing the parameter values necessary for the mass
            calculations.

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
        parameter_set : dict, optional
            A dictionary containing the parameter values necessary for the volume
            calculation.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the subclass.
        """
        raise NotImplementedError

    def approximate_capacity(self, inputs: Inputs):
        """
        Calculate a new estimate for the nominal capacity based on the theoretical energy density
        and an average voltage.

        This method must be implemented by subclasses.

        Parameters
        ----------
        inputs : Inputs
            The parameters that are the inputs of the model.

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
