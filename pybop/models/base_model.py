from dataclasses import dataclass
from typing import Any, Dict
import pybamm
import numpy as np
import casadi

Inputs = Dict[str, float]


@dataclass
class TimeSeriesState(object):
    """
    The current state of a time series model that is a pybamm model
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


class BaseModel:
    """
    Base class for pybop models.
    """

    def __init__(self, name="Base Model"):
        self.name = name
        self.pybamm_model = None
        self.parameters = None
        self.dataset = None
        self.signal = None

    def build(
        self,
        dataset=None,
        parameters=None,
        check_model=True,
        init_soc=None,
    ):
        """
        Build the PyBOP model (if not built already).
        For PyBaMM forward models, this method follows a
        similar process to pybamm.Simulation.build().
        """
        self.dataset = dataset
        self.parameters = parameters
        if self.parameters is not None:
            self.fit_keys = [param.name for param in self.parameters]

        if init_soc is not None:
            self.set_init_soc(init_soc)

        if self._built_model:
            return

        elif self.pybamm_model.is_discretised:
            self._model_with_set_params = self.pybamm_model
            self._built_model = self.pybamm_model
        else:
            self.set_params()

            any_disc_variables_exist = (
                self.geometry is not None
                or self.var_pts is not None
                or self.spatial_methods is not None
                or self.submesh_types is not None
            )
            if any_disc_variables_exist and self.geometry is None:
                raise ValueError(
                    "Geometry must be provided if any discretisation variables are provided"
                )
            if any_disc_variables_exist and self.submesh_types is None:
                raise ValueError(
                    "Submesh types must be provided if any discretisation variables are provided"
                )
            if any_disc_variables_exist and self.var_pts is None:
                raise ValueError(
                    "Variable points must be provided if any discretisation variables are provided"
                )
            if any_disc_variables_exist and self.spatial_methods is None:
                raise ValueError(
                    "Spatial methods must be provided if any discretisation variables are provided"
                )

            if any_disc_variables_exist:
                self._mesh = pybamm.Mesh(
                    self.geometry, self.submesh_types, self.var_pts
                )
                self._disc = pybamm.Discretisation(self.mesh, self.spatial_methods)
                self._built_model = self._disc.process_model(
                    self._model_with_set_params, inplace=False, check_model=check_model
                )
            else:
                self._built_model = self._model_with_set_params

            # Clear solver and setup model
            self._solver._model_set_up = {}

    def set_init_soc(self, init_soc):
        """
        Set the initial state of charge.
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

    def set_params(self):
        """
        Set the parameters in the model.
        """
        if self.model_with_set_params:
            return

        # Mark any simulation inputs in the parameter set
        if self.parameters is not None:
            for i in self.fit_keys:
                self._parameter_set[i] = "[input]"

        if self.dataset is not None and self.parameters is not None:
            if "Current function [A]" not in self.fit_keys:
                self.parameter_set["Current function [A]"] = pybamm.Interpolant(
                    self.dataset["Time [s]"].data,
                    self.dataset["Current function [A]"].data,
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

    def reinit(
        self, inputs: Inputs, t: float = 0.0, x: np.ndarray | None = None
    ) -> TimeSeriesState:
        """
        Returns the initial state of the problem.
        """
        if self._built_model is None:
            raise ValueError("Model must be built before calling reinit")

        self._solver.set_up(self._built_model, inputs=inputs)

        if not isinstance(inputs, dict):
            inputs = {key: inputs[i] for i, key in enumerate(self.fit_keys)}

        if x is None:
            x = self._built_model.y0

        sol = pybamm.Solution([np.array([t])], [x], self._built_model, inputs)

        return TimeSeriesState(sol=sol, inputs=inputs, t=t)

    def step(self, state: TimeSeriesState, time: np.ndarray) -> TimeSeriesState:
        """
        step forward in time from the given state until the given time.

        Parameters
        ----------
        state : TimeSeriesState
            The current state of the model
        time : np.ndarray
            The time to predict the system to
        """
        dt = time - state.t
        new_sol = self.solver.step(
            state.sol, self.built_model, dt, npts=2, inputs=state.inputs, save=False
        )
        return TimeSeriesState(sol=new_sol, inputs=state.inputs, t=time)

    def simulate(self, inputs, t_eval) -> np.ndarray[np.float64]:
        """
        Run the forward model and return the result in Numpy array format
        aligning with Pints' ForwardModel simulate method.
        """

        if self._built_model is None:
            raise ValueError("Model must be built before calling simulate")
        else:
            if not isinstance(inputs, dict):
                inputs = {key: inputs[i] for i, key in enumerate(self.fit_keys)}

            return self.solver.solve(self.built_model, inputs=inputs, t_eval=t_eval)[
                self.signal
            ].data

    def simulateS1(self, inputs, t_eval):
        """
        Run the forward model and return the function evaulation and it's gradient
        aligning with Pints' ForwardModel simulateS1 method.
        """

        if self._built_model is None:
            raise ValueError("Model must be built before calling simulate")
        else:
            if not isinstance(inputs, dict):
                inputs = {key: inputs[i] for i, key in enumerate(self.fit_keys)}

            sol = self.solver.solve(
                self.built_model,
                inputs=inputs,
                t_eval=t_eval,
                calculate_sensitivities=True,
            )

            return (
                sol[self.signal].data,
                np.asarray(
                    [
                        sol[self.signal].sensitivities[key].toarray()
                        for key in self.fit_keys
                    ]
                ).T,
            )

    def predict(
        self,
        inputs=None,
        t_eval=None,
        parameter_set=None,
        experiment=None,
        init_soc=None,
    ):
        """
        Create a PyBaMM simulation object, solve it, and return a solution object.
        """
        parameter_set = parameter_set or self._parameter_set
        if inputs is not None:
            if not isinstance(inputs, dict):
                inputs = {key: inputs[i] for i, key in enumerate(self.fit_keys)}
            parameter_set.update(inputs)

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
            raise ValueError("This sim method currently only supports PyBaMM models")

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
    def geometry(self, geometry: pybamm.Geometry | None):
        self._geometry = geometry.copy() if geometry is not None else None

    @property
    def submesh_types(self):
        return self._submesh_types

    @submesh_types.setter
    def submesh_types(self, submesh_types: Dict[str, Any] | None):
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
    def var_pts(self, var_pts: Dict[str, int] | None):
        self._var_pts = var_pts.copy() if var_pts is not None else None

    @property
    def spatial_methods(self):
        return self._spatial_methods

    @spatial_methods.setter
    def spatial_methods(self, spatial_methods: Dict[str, Any] | None):
        self._spatial_methods = spatial_methods.copy() if spatial_methods else None

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, solver):
        self._solver = solver.copy() if solver is not None else None
