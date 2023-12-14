import pybamm
import numpy as np


class BaseModel:
    """
    A base class for constructing and simulating models using PyBaMM.

    This class serves as a foundation for building specific models in PyBaMM.
    It provides methods to set up the model, define parameters, and perform
    simulations. The class is designed to be subclassed for creating models
    with custom behavior.

    """

    def __init__(self, name="Base Model"):
        """
        Initialize the BaseModel with an optional name.

        Parameters
        ----------
        name : str, optional
            The name given to the model instance.
        """
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
        Construct the PyBaMM model if not already built, and set parameters.

        This method initializes the model components, applies the given parameters,
        sets up the mesh and discretization if needed, and prepares the model
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
            self._mesh = pybamm.Mesh(self.geometry, self.submesh_types, self.var_pts)
            self._disc = pybamm.Discretisation(self.mesh, self.spatial_methods)
            self._built_model = self._disc.process_model(
                self._model_with_set_params, inplace=False, check_model=check_model
            )

            # Clear solver
            self._solver._model_set_up = {}

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

    def set_params(self):
        """
        Assign the parameters to the model.

        This method processes the model with the given parameters, sets up
        the geometry, and updates the model instance.
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
                    self.dataset["Time [s]"],
                    self.dataset["Current function [A]"],
                    pybamm.t,
                )
                # Set t_eval
                self.time_data = self._parameter_set["Current function [A]"].x[0]

        self._model_with_set_params = self._parameter_set.process_model(
            self._unprocessed_model, inplace=False
        )
        self._parameter_set.process_geometry(self.geometry)
        self.pybamm_model = self._model_with_set_params

    def simulate(self, inputs, t_eval):
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
            if not isinstance(inputs, dict):
                inputs = {key: inputs[i] for i, key in enumerate(self.fit_keys)}

            sol = self.solver.solve(self.built_model, inputs=inputs, t_eval=t_eval)

            predictions = [sol[signal].data for signal in self.signal]

            return np.vstack(predictions).T

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
            if not isinstance(inputs, dict):
                inputs = {key: inputs[i] for i, key in enumerate(self.fit_keys)}

            sol = self.solver.solve(
                self.built_model,
                inputs=inputs,
                t_eval=t_eval,
                calculate_sensitivities=True,
            )

            predictions = [sol[signal].data for signal in self.signal]

            sensitivities = [
                np.array(
                    [[sol[signal].sensitivities[key]] for signal in self.signal]
                ).reshape(len(sol[self.signal[0]].data), self.n_outputs)
                for key in self.fit_keys
            ]

            return np.vstack(predictions).T, np.dstack(sensitivities)

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
    def geometry(self, geometry):
        self._geometry = geometry.copy()

    @property
    def submesh_types(self):
        return self._submesh_types

    @submesh_types.setter
    def submesh_types(self, submesh_types):
        self._submesh_types = submesh_types.copy()

    @property
    def mesh(self):
        return self._mesh

    @property
    def var_pts(self):
        return self._var_pts

    @var_pts.setter
    def var_pts(self, var_pts):
        self._var_pts = var_pts.copy()

    @property
    def spatial_methods(self):
        return self._spatial_methods

    @spatial_methods.setter
    def spatial_methods(self, spatial_methods):
        self._spatial_methods = spatial_methods.copy()

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, solver):
        self._solver = solver.copy()
