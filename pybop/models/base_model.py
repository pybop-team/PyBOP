import copy
from dataclasses import dataclass
from typing import Any, Optional, Union

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

    This class serves as a foundation for building specific models in PyBaMM. It provides
    methods to set up the model, define parameters, and perform simulations. The class is
    designed to be subclassed for creating models with custom behaviour.

    """

    def __init__(
        self, name: str = "Base Model", parameter_set: Optional[ParameterSet] = None
    ):
        """
        Initialize the BaseModel with an optional name and a parameter set.

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
            self.parameter_set = pybamm.ParameterValues(parameter_set)
        elif isinstance(parameter_set, pybamm.ParameterValues):
            self.parameter_set = parameter_set
        else:  # a pybop parameter set
            self.parameter_set = pybamm.ParameterValues(parameter_set.params)

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
        initial_state: Optional[float] = None,
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
        initial_state : float or str, optional
            If float, this value is used as the initial state of charge (as a decimal between 0
            and 1). If str ending in "V", this value is used as the initial open-circuit voltage.
            Defaults to None, indicating that the existing initial concentrations will be used.
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
            if not self.pybamm_model._built:
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
            self._solver._model_set_up = {}

        self.n_states = self._built_model.len_rhs_and_alg  # len_rhs + len_alg

    def set_initial_state(self, initial_state: Optional[float] = None):
        """
        Set the initial concentrations for the battery model.

        Parameters
        ----------
        initial_state : float or str, optional
            If float, this value is used as the initial state of charge (as a decimal between 0
            and 1). If str ending in "V", this value is used as the initial open-circuit voltage.
            Defaults to None, indicating that the existing initial concentrations will be used.
        """
        if "Initial SoC" in self._unprocessed_parameter_set.keys():
            initial_soc = self.get_initial_state(initial_state)
            self._unprocessed_parameter_set.update({"Initial SoC": initial_soc})

        else:
            # Point pybamm variables at pybop variables
            self.model = self.pybamm_model
            self._model = self.pybamm_model
            self._unprocessed_parameter_values = self._unprocessed_parameter_set

            pybamm_set_initial_state = pybamm.Simulation.set_initial_soc
            pybamm_set_initial_state(self, initial_state, inputs=None)

            # Update the default parameter set for consistency
            self._unprocessed_parameter_set = self._parameter_values

            # Clear the pybamm variables
            del self.model
            del self._model
            del self._unprocessed_parameter_values
            del self._parameter_values

        # Use a copy of the updated default parameter set
        self.parameter_set = self._unprocessed_parameter_set

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

    def rebuild(
        self,
        dataset: Optional[Dataset] = None,
        parameters: Union[Parameters, dict] = None,
        check_model: bool = True,
        initial_state: Optional[float] = None,
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
        initial_state : float or str, optional
            If float, this value is used as the initial state of charge. If str ending in "V", this
            value is used as the initial open-circuit voltage. Defaults to None, indicating that the
            initial concentrations in the parameter set should be used.
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
        self._solver._model_set_up = {}

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
            self.geometry = self.pybamm_model.default_geometry

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
        self, inputs: Inputs, t_eval: np.array, initial_state: Optional[float] = None
    ):
        """
        Execute the forward model simulation and return the result.

        Parameters
        ----------
        inputs : Inputs
            The input parameters for the simulation.
        t_eval : array-like
            An array of time points at which to evaluate the solution.
        initial_state : float or str, optional
            If float, this value is used as the initial state of charge (as a decimal between 0
            and 1). If str ending in "V", this value is used as the initial open-circuit voltage.
            Defaults to None, indicating that the existing initial concentrations will be used.

        Returns
        -------
        pybamm.Solution or [np.inf]
            The solution object returned by a PyBaMM simulation, or [np.inf] in the case where
            the parameter values are infeasible and infeasible solutions are not allowed.

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

        if self.check_params(
            inputs=inputs,
            allow_infeasible_solutions=self.allow_infeasible_solutions,
        ):
            try:
                return self.solver.solve(
                    self._built_model, inputs=inputs, t_eval=t_eval
                )
            except Exception as e:
                print(f"Error: {e}")
                return [np.inf]
        else:
            return [np.inf]

    def simulateS1(
        self, inputs: Inputs, t_eval: np.array, initial_state: Optional[float] = None
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
        initial_state : float or str, optional
            If float, this value is used as the initial state of charge (as a decimal between 0
            and 1). If str ending in "V", this value is used as the initial open-circuit voltage.
            Defaults to None, indicating that the existing initial concentrations will be used.

        Returns
        -------
        pybamm.Solution or [np.inf]
            The solution object returned by a PyBaMM simulation, or [np.inf] in the case where
            the parameter values are infeasible and infeasible solutions are not allowed.

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

        if self.check_params(
            inputs=inputs,
            allow_infeasible_solutions=self.allow_infeasible_solutions,
        ):
            try:
                return self._solver.solve(
                    self._built_model,
                    inputs=inputs,
                    t_eval=t_eval,
                    calculate_sensitivities=True,
                )

            except Exception as e:
                print(f"Error: {e}")
                return [np.inf]
        else:
            return [np.inf]

    def predict(
        self,
        inputs: Optional[Inputs] = None,
        t_eval: Optional[np.array] = None,
        parameter_set: Optional[ParameterSet] = None,
        experiment: Optional[Experiment] = None,
        initial_state: Optional[float] = None,
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
        initial_state : float or str, optional
            If float, this value is used as the initial state of charge (as a decimal between 0
            and 1). If str ending in "V", this value is used as the initial open-circuit voltage.
            Defaults to None, indicating that the existing initial concentrations will be used.

        Returns
        -------
        pybamm.Solution or [np.inf]
            The solution object returned by a PyBaMM simulation, or [np.inf] in the case where
            the parameter values are infeasible and infeasible solutions are not allowed.

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
        elif not self._unprocessed_model._built:
            self._unprocessed_model.build_model()

        parameter_set = parameter_set or self._unprocessed_parameter_set.copy()
        if inputs is not None:
            inputs = self.parameters.verify(inputs)
            parameter_set.update(inputs)

        if initial_state is not None:
            if isinstance(self.pybamm_model, pybamm.equivalent_circuit.Thevenin):
                parameter_set["Initial SoC"] = initial_state
                initial_state = None
            else:
                # Update the default initial state just for consistency
                self.set_initial_state(initial_state)

        if self.check_params(
            parameter_set=parameter_set,
            allow_infeasible_solutions=self.allow_infeasible_solutions,
        ):
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
                    "The predict method requires either an experiment or "
                    "t_eval to be specified."
                )

        else:
            return [np.inf]

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
        parameter_set : pybop.parameter_set, optional
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
        parameter_set : pybop.parameter_set
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
    def submesh_types(self, submesh_types: Optional[dict[str, Any]]):
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
    def var_pts(self, var_pts: Optional[dict[str, int]]):
        self._var_pts = var_pts.copy() if var_pts is not None else None

    @property
    def spatial_methods(self):
        return self._spatial_methods

    @spatial_methods.setter
    def spatial_methods(self, spatial_methods: Optional[dict[str, Any]]):
        self._spatial_methods = (
            spatial_methods.copy() if spatial_methods is not None else None
        )

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, solver):
        self._solver = solver.copy() if solver is not None else None

    def get_parameter_info(self, print_info: bool = False):
        """
        Extracts the parameter names and types and returns them as a dictionary.
        """
        if not self.pybamm_model._built:
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
