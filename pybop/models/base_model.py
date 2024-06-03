import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional

import casadi
import numpy as np
import pybamm

Inputs = Dict[str, float]


@dataclass
class TimeSeriesState(object):
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

    This class serves as a foundation for building specific models in PyBaMM.
    It provides methods to set up the model, define parameters, and perform
    simulations. The class is designed to be subclassed for creating models
    with custom behaviour.

    """

    def __init__(self, name="Base Model", parameter_set=None):
        """
        Initialize the BaseModel with an optional name.

        Parameters
        ----------
        name : str, optional
            The name given to the model instance.
        """
        self.name = name
        if parameter_set is None:
            self._parameter_set = None
        elif isinstance(parameter_set, dict):
            self._parameter_set = pybamm.ParameterValues(parameter_set)
        elif isinstance(parameter_set, pybamm.ParameterValues):
            self._parameter_set = parameter_set
        else:  # a pybop parameter set
            self._parameter_set = pybamm.ParameterValues(parameter_set.params)

        self.pybamm_model = None
        self.parameters = None
        self.dataset = None
        self.signal = None
        self.additional_variables = []
        self.matched_parameters = {}
        self.non_matched_parameters = {}
        self.fit_keys = []
        self.param_check_counter = 0
        self.allow_infeasible_solutions = True

    def build(
        self,
        dataset=None,
        parameters=None,
        check_model=True,
        init_soc=None,
    ):
        """
        Construct the PyBaMM model if not already built, and set parameters.

        This method initializes the model components, applies the given parameters,
        sets up the mesh and discretisation if needed, and prepares the model
        for simulations.

        Parameters
        ----------
        dataset : pybamm.Dataset, optional
            The dataset to be used in the model construction.
        parameters : dict, optional
            A dictionary containing parameter values to apply to the model.
        check_model : bool, optional
            If True, the model will be checked for correctness after construction.
        init_soc : float, optional
            The initial state of charge to be used in simulations.
        """
        self.dataset = dataset
        self.parameters = parameters
        if self.parameters is not None:
            self.classify_and_update_parameters(self.parameters)
            self.fit_keys = [param.name for param in self.parameters]
        else:
            self.fit_keys = []

        if init_soc is not None:
            self.set_init_soc(init_soc)

        if self._built_model:
            return

        elif self.pybamm_model.is_discretised:
            self._model_with_set_params = self.pybamm_model
            self._built_model = self.pybamm_model

        else:
            if not self.pybamm_model._built:
                self.pybamm_model.build_model()
            self.set_params()

            self._mesh = pybamm.Mesh(self.geometry, self.submesh_types, self.var_pts)
            self._disc = pybamm.Discretisation(self.mesh, self.spatial_methods)
            self._built_model = self._disc.process_model(
                self._model_with_set_params, inplace=False, check_model=check_model
            )

            # Clear solver and setup model
            self._solver._model_set_up = {}

        self.n_states = self._built_model.len_rhs_and_alg  # len_rhs + len_alg

    def set_init_soc(self, init_soc):
        """
        Set the initial state of charge for the battery model.

        Parameters
        ----------
        init_soc : float
            The initial state of charge to be used in the model.
        """
        if self._built_initial_soc != init_soc:
            # reset
            self._model_with_set_params = None
            self._built_model = None
            self.op_conds_to_built_models = None
            self.op_conds_to_built_solvers = None

        param = self.pybamm_model.param
        self._parameter_set = (
            self._unprocessed_parameter_set.set_initial_stoichiometries(
                init_soc, param=param, inplace=False
            )
        )
        # Save solved initial SOC in case we need to rebuild the model
        self._built_initial_soc = init_soc

    def set_params(self, rebuild=False):
        """
        Assign the parameters to the model.

        This method processes the model with the given parameters, sets up
        the geometry, and updates the model instance.
        """
        if self.model_with_set_params and not rebuild:
            return

        # Mark any simulation inputs in the parameter set
        for key in self.non_matched_parameters.keys():
            self._parameter_set[key] = "[input]"

        if self.dataset is not None and (not self.matched_parameters or not rebuild):
            if "Current function [A]" not in self.fit_keys:
                self._parameter_set["Current function [A]"] = pybamm.Interpolant(
                    self.dataset["Time [s]"],
                    self.dataset["Current function [A]"],
                    pybamm.t,
                )
                # Set t_eval
                self.time_data = self._parameter_set["Current function [A]"].x[0]

        self._model_with_set_params = self._parameter_set.process_model(
            self._unprocessed_model, inplace=False
        )
        if self.geometry is not None:
            self._parameter_set.process_geometry(self.geometry)
        self.pybamm_model = self._model_with_set_params

    def rebuild(
        self,
        dataset=None,
        parameters=None,
        parameter_set=None,
        check_model=True,
        init_soc=None,
    ):
        """
        Rebuild the PyBaMM model for a given parameter set.

        This method requires the self.build() method to be called first, and
        then rebuilds the model for a given parameter set. Specifically,
        this method applies the given parameters, sets up the mesh and
        discretisation if needed, and prepares the model for simulations.

        Parameters
        ----------
        dataset : pybamm.Dataset, optional
            The dataset to be used in the model construction.
        parameters : dict, optional
            A dictionary containing parameter values to apply to the model.
        parameter_set : pybop.parameter_set, optional
            A PyBOP parameter set object or a dictionary containing the parameter values
        check_model : bool, optional
            If True, the model will be checked for correctness after construction.
        init_soc : float, optional
            The initial state of charge to be used in simulations.
        """
        self.dataset = dataset
        self.parameters = parameters
        if parameters is not None:
            self.classify_and_update_parameters(parameters)

        if init_soc is not None:
            self.set_init_soc(init_soc)

        if self._built_model is None:
            raise ValueError("Model must be built before calling rebuild")

        self.set_params(rebuild=True)
        self._mesh = pybamm.Mesh(self.geometry, self.submesh_types, self.var_pts)
        self._disc = pybamm.Discretisation(self.mesh, self.spatial_methods)
        self._built_model = self._disc.process_model(
            self._model_with_set_params, inplace=False, check_model=check_model
        )

        # Clear solver and setup model
        self._solver._model_set_up = {}

    def classify_and_update_parameters(self, parameters):
        """
        Update the parameter values according to their classification as either
        'matched_parameters' which require a model rebuild and
        'non_matched_parameters' which are standard inputs.

        Parameters
        ----------
        parameters : pybop.ParameterSet

        """
        parameter_dictionary = {param.name: param.value for param in parameters}
        matched_parameters = {
            param: parameter_dictionary[param]
            for param in parameter_dictionary
            if param in self.rebuild_parameters
        }
        non_matched_parameters = {
            param: parameter_dictionary[param]
            for param in parameter_dictionary
            if param not in self.rebuild_parameters
        }

        self.matched_parameters.update(matched_parameters)
        self.non_matched_parameters.update(non_matched_parameters)

        if self.matched_parameters:
            self._parameter_set.update(self.matched_parameters)
            self._unprocessed_parameter_set = self._parameter_set
            self.geometry = self.pybamm_model.default_geometry

    def reinit(
        self, inputs: Inputs, t: float = 0.0, x: Optional[np.ndarray] = None
    ) -> TimeSeriesState:
        """
        Initialises the solver with the given inputs and returns the initial state of the problem
        """
        if self._built_model is None:
            raise ValueError("Model must be built before calling reinit")

        if not isinstance(inputs, dict):
            inputs = {key: inputs[i] for i, key in enumerate(self.fit_keys)}

        self._solver.set_up(self._built_model, inputs=inputs)

        if x is None:
            x = self._built_model.y0

        sol = pybamm.Solution([np.array([t])], [x], self._built_model, inputs)

        return TimeSeriesState(sol=sol, inputs=inputs, t=t)

    def get_state(self, inputs: Inputs, t: float, x: np.ndarray) -> TimeSeriesState:
        """
        Returns the given state for the problem (inputs are assumed constant since last reinit)
        """
        if self._built_model is None:
            raise ValueError("Model must be built before calling get_state")

        sol = pybamm.Solution([np.array([t])], [x], self._built_model, inputs)

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
            state.sol, self.built_model, dt, npts=2, inputs=state.inputs, save=False
        )
        return TimeSeriesState(sol=new_sol, inputs=state.inputs, t=time)

    def simulate(self, inputs, t_eval) -> np.ndarray[np.float64]:
        """
        Execute the forward model simulation and return the result.

        Parameters
        ----------
        inputs : dict or array-like
            The input parameters for the simulation. If array-like, it will be
            converted to a dictionary using the model's fit keys.
        t_eval : array-like
            An array of time points at which to evaluate the solution.

        Returns
        -------
        array-like
            The simulation result corresponding to the specified signal.

        Raises
        ------
        ValueError
            If the model has not been built before simulation.
        """
        if self._built_model is None:
            raise ValueError("Model must be built before calling simulate")
        else:
            if self.matched_parameters and not self.non_matched_parameters:
                sol = self.solver.solve(self.built_model, t_eval=t_eval)

            else:
                if not isinstance(inputs, dict):
                    inputs = {key: inputs[i] for i, key in enumerate(self.fit_keys)}

                if self.check_params(
                    inputs=inputs,
                    allow_infeasible_solutions=self.allow_infeasible_solutions,
                ):
                    try:
                        sol = self.solver.solve(
                            self.built_model, inputs=inputs, t_eval=t_eval
                        )
                    except Exception as e:
                        print(f"Error: {e}")
                        return {signal: [np.inf] for signal in self.signal}
                else:
                    return {signal: [np.inf] for signal in self.signal}

            y = {
                signal: sol[signal].data
                for signal in (self.signal + self.additional_variables)
            }

            return y

    def simulateS1(self, inputs, t_eval):
        """
        Perform the forward model simulation with sensitivities.

        Parameters
        ----------
        inputs : dict or array-like
            The input parameters for the simulation. If array-like, it will be
            converted to a dictionary using the model's fit keys.
        t_eval : array-like
            An array of time points at which to evaluate the solution and its
            sensitivities.

        Returns
        -------
        tuple
            A tuple containing the simulation result and the sensitivities.

        Raises
        ------
        ValueError
            If the model has not been built before simulation.
        """

        if self._built_model is None:
            raise ValueError("Model must be built before calling simulate")
        else:
            if self.matched_parameters:
                raise ValueError(
                    "Cannot use sensitivies for parameters which require a model rebuild"
                )

            if not isinstance(inputs, dict):
                inputs = {key: inputs[i] for i, key in enumerate(self.fit_keys)}

            if self.check_params(
                inputs=inputs,
                allow_infeasible_solutions=self.allow_infeasible_solutions,
            ):
                try:
                    sol = self._solver.solve(
                        self.built_model,
                        inputs=inputs,
                        t_eval=t_eval,
                        calculate_sensitivities=True,
                    )
                    y = {signal: sol[signal].data for signal in self.signal}

                    # Extract the sensitivities and stack them along a new axis for each signal
                    dy = np.empty(
                        (
                            sol[self.signal[0]].data.shape[0],
                            self.n_outputs,
                            self.n_parameters,
                        )
                    )

                    for i, signal in enumerate(self.signal):
                        dy[:, i, :] = np.stack(
                            [
                                sol[signal].sensitivities[key].toarray()[:, 0]
                                for key in self.fit_keys
                            ],
                            axis=-1,
                        )

                    return y, dy
                except Exception as e:
                    print(f"Error: {e}")
                    return {signal: [np.inf] for signal in self.signal}, [np.inf]

            else:
                return {signal: [np.inf] for signal in self.signal}, [np.inf]

    def predict(
        self,
        inputs=None,
        t_eval=None,
        parameter_set=None,
        experiment=None,
        init_soc=None,
    ):
        """
        Solve the model using PyBaMM's simulation framework and return the solution.

        This method sets up a PyBaMM simulation by configuring the model, parameters, experiment
        (if any), and initial state of charge (if provided). It then solves the simulation and
        returns the resulting solution object.

        Parameters
        ----------
        inputs : dict or array-like, optional
            Input parameters for the simulation. If the input is array-like, it is converted
            to a dictionary using the model's fitting keys. Defaults to None, indicating
            that the default parameters should be used.
        t_eval : array-like, optional
            An array of time points at which to evaluate the solution. Defaults to None,
            which means the time points need to be specified within experiment or elsewhere.
        parameter_set : pybamm.ParameterValues, optional
            A PyBaMM ParameterValues object or a dictionary containing the parameter values
            to use for the simulation. Defaults to the model's current ParameterValues if None.
        experiment : pybamm.Experiment, optional
            A PyBaMM Experiment object specifying the experimental conditions under which
            the simulation should be run. Defaults to None, indicating no experiment.
        init_soc : float, optional
            The initial state of charge for the simulation, as a fraction (between 0 and 1).
            Defaults to None.

        Returns
        -------
        pybamm.Solution
            The solution object returned after solving the simulation.

        Raises
        ------
        ValueError
            If the model has not been configured properly before calling this method or
            if PyBaMM models are not supported by the current simulation method.

        """
        if not self.pybamm_model._built:
            self.pybamm_model.build_model()

        parameter_set = parameter_set or self._unprocessed_parameter_set
        if inputs is not None:
            if not isinstance(inputs, dict):
                inputs = {key: inputs[i] for i, key in enumerate(self.fit_keys)}
            parameter_set.update(inputs)

        if self.check_params(
            inputs=inputs,
            parameter_set=parameter_set,
            allow_infeasible_solutions=self.allow_infeasible_solutions,
        ):
            if self._unprocessed_model is not None:
                if experiment is None:
                    return pybamm.Simulation(
                        self._unprocessed_model,
                        parameter_values=parameter_set,
                    ).solve(t_eval=t_eval, initial_soc=init_soc)
                else:
                    return pybamm.Simulation(
                        self._unprocessed_model,
                        experiment=experiment,
                        parameter_values=parameter_set,
                    ).solve(initial_soc=init_soc)
            else:
                raise ValueError(
                    "This sim method currently only supports PyBaMM models"
                )

        else:
            return [np.inf]

    def check_params(
        self, inputs=None, parameter_set=None, allow_infeasible_solutions=True
    ):
        """
        Check compatibility of the model parameters.

        Parameters
        ----------
        inputs : dict
            The input parameters for the simulation.
        allow_infeasible_solutions : bool, optional
            If True, infeasible parameter values will be allowed in the optimisation (default: True).

        Returns
        -------
        bool
            A boolean which signifies whether the parameters are compatible.

        """
        if inputs is not None:
            if not isinstance(inputs, dict):
                if isinstance(inputs, list):
                    for entry in inputs:
                        if not isinstance(entry, (int, float)):
                            raise ValueError(
                                "Expecting inputs in the form of a dictionary, numeric list"
                                + f" or None, but received a list with type: {type(inputs)}"
                            )
                else:
                    inputs = {key: inputs[i] for i, key in enumerate(self.fit_keys)}

        return self._check_params(
            inputs=inputs, allow_infeasible_solutions=allow_infeasible_solutions
        )

    def _check_params(self, inputs=None, allow_infeasible_solutions=True):
        """
        A compatibility check for the model parameters which can be implemented by subclasses
        if required, otherwise it returns True by default.

        Parameters
        ----------
        inputs : dict
            The input parameters for the simulation.
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

    def cell_mass(self, parameter_set=None):
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

    def cell_volume(self, parameter_set=None):
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

    def approximate_capacity(self, x):
        """
        Calculate a new estimate for the nominal capacity based on the theoretical energy density
        and an average voltage.

        This method must be implemented by subclasses.

        Parameters
        ----------
        x : array-like
            An array of values representing the model inputs.

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
    def parameter_set(self):
        return self._parameter_set

    @parameter_set.setter
    def parameter_set(self, parameter_set):
        self._parameter_set = parameter_set.copy()

    @property
    def model_with_set_params(self):
        return self._model_with_set_params

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, geometry: Optional[pybamm.Geometry]):
        self._geometry = geometry.copy() if geometry is not None else None

    @property
    def submesh_types(self):
        return self._submesh_types

    @submesh_types.setter
    def submesh_types(self, submesh_types: Optional[Dict[str, Any]]):
        self._submesh_types = (
            submesh_types.copy() if submesh_types is not None else None
        )

    @property
    def mesh(self):
        return self._mesh

    @property
    def var_pts(self):
        return self._var_pts

    @var_pts.setter
    def var_pts(self, var_pts: Optional[Dict[str, int]]):
        self._var_pts = var_pts.copy() if var_pts is not None else None

    @property
    def spatial_methods(self):
        return self._spatial_methods

    @spatial_methods.setter
    def spatial_methods(self, spatial_methods: Optional[Dict[str, Any]]):
        self._spatial_methods = (
            spatial_methods.copy() if spatial_methods is not None else None
        )

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, solver):
        self._solver = solver.copy() if solver is not None else None
