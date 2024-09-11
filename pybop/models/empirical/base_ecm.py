import numpy as np
import pybamm

from pybop.models.base_model import BaseModel, Inputs
from pybop.parameters.parameter_set import ParameterSet


class ECircuitModel(BaseModel):
    """
    Overwrites and extends `BaseModel` class for circuit-based PyBaMM models.

    Parameters
    ----------
    pybamm_model : pybamm.BaseModel
        A subclass of the pybamm Base Model.
    name : str, optional
        The name for the model instance, defaulting to "Empirical Base Model".
    parameter_set : pybamm.ParameterValues or dict, optional
        The parameters for the model. If None, default parameters provided by PyBaMM are used.
    geometry : dict, optional
        The geometry definitions for the model. If None, default geometry from PyBaMM is used.
    submesh_types : dict, optional
        The types of submeshes to use. If None, default submesh types from PyBaMM are used.
    var_pts : dict, optional
        The discretization points for each variable in the model. If None, default points from PyBaMM are used.
    spatial_methods : dict, optional
        The spatial methods used for discretization. If None, default spatial methods from PyBaMM are used.
    solver : pybamm.Solver, optional
        The solver to use for simulating the model. If None, the default solver from PyBaMM is used.
    **model_kwargs : optional
        Valid PyBaMM model option keys and their values. For example,
        build : bool, optional
            If True, the model is built upon creation (default: False).
        options : dict, optional
            A dictionary of options to customise the behaviour of the PyBaMM model.
    """

    def __init__(
        self,
        pybamm_model,
        name="Empirical Base Model",
        parameter_set=None,
        geometry=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
        check_params=None,
        eis=False,
        **model_kwargs,
    ):
        model_options = dict(build=False)
        for key, value in model_kwargs.items():
            model_options[key] = value
        pybamm_model = pybamm_model(**model_options)

        # Correct OCP if set to default
        if (
            parameter_set is not None
            and "Open-circuit voltage [V]" in parameter_set.keys()
        ):
            default_ocp = pybamm_model.default_parameter_values[
                "Open-circuit voltage [V]"
            ]
            if parameter_set["Open-circuit voltage [V]"] == "default":
                print("Setting open-circuit voltage to default function")
                parameter_set["Open-circuit voltage [V]"] = default_ocp

        super().__init__(
            name=name, parameter_set=parameter_set, check_params=check_params, eis=eis
        )
        self.pybamm_model = pybamm_model
        self._unprocessed_model = self.pybamm_model

        # Set parameters, using either the provided ones or the default
        self.default_parameter_values = self.pybamm_model.default_parameter_values
        self._parameter_set = self._parameter_set or self.default_parameter_values
        self._unprocessed_parameter_set = self._parameter_set

        # Define model geometry and discretization
        self._geometry = geometry or self.pybamm_model.default_geometry
        self._submesh_types = submesh_types or self.pybamm_model.default_submesh_types
        self._var_pts = var_pts or self.pybamm_model.default_var_pts
        self._spatial_methods = (
            spatial_methods or self.pybamm_model.default_spatial_methods
        )
        self._solver = solver or self.pybamm_model.default_solver

        # Internal attributes for the built model are initialized but not set
        self._model_with_set_params = None
        self._built_model = None
        self._built_initial_soc = None
        self._mesh = None
        self._disc = None
        self.geometric_parameters = {}

    def _check_params(
        self,
        inputs: Inputs,
        parameter_set: ParameterSet,
        allow_infeasible_solutions: bool = True,
    ):
        """
        Check the compatibility of the model parameters.

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
        if self.param_checker:
            return self.param_checker(inputs, allow_infeasible_solutions)
        return True

    def get_initial_state(
        self,
        initial_value,
        parameter_values=None,
        param=None,
        options=None,
        tol=1e-6,
        inputs=None,
    ):
        """
        Calculate the initial state of charge given an open-circuit voltage, voltage limits
        and the open-circuit voltage function defined by the parameter set.

        Parameters
        ----------
        initial_value : float
            Target initial value.
            If float, interpreted as SOC, must be between 0 and 1.
            If string e.g. "4 V", interpreted as voltage, must be between V_min and V_max.
        parameter_values : :class:`pybamm.ParameterValues`
            The parameter values class that will be used for the simulation. Required for
            calculating appropriate initial stoichiometries.
        param : :class:`pybamm.LithiumIonParameters`, optional
            The symbolic parameter set to use for the simulation.
            If not provided, the default parameter set will be used.
        options : dict-like, optional
            A dictionary of options to be passed to the model, see
            :class:`pybamm.BatteryModelOptions`.
        tol : float, optional
            The tolerance for the solver used to compute the initial stoichiometries.
            A lower value results in higher precision but may increase computation time.
            Default is 1e-6.

        Returns
        -------
        initial_soc
            The initial state of charge
        """
        parameter_values = parameter_values or self._unprocessed_parameter_set
        param = param or self.pybamm_model.param

        if isinstance(initial_value, str) and initial_value.endswith("V"):
            V_init = float(initial_value[:-1])
            V_min = parameter_values.evaluate(param.voltage_low_cut, inputs=inputs)
            V_max = parameter_values.evaluate(param.voltage_high_cut, inputs=inputs)

            if not V_min <= V_init <= V_max:
                raise ValueError(
                    f"Initial voltage {V_init}V is outside the voltage limits "
                    f"({V_min}, {V_max})"
                )

            # Solve simple model for initial soc based on target voltage
            soc_model = pybamm.BaseModel()
            soc = pybamm.Variable("soc")
            ocv = param.ocv
            soc_model.algebraic[soc] = ocv(soc) - V_init

            # initial guess for soc linearly interpolates between 0 and 1
            # based on V linearly interpolating between V_max and V_min
            soc_model.initial_conditions[soc] = (V_init - V_min) / (V_max - V_min)
            soc_model.variables["soc"] = soc
            parameter_values.process_model(soc_model)
            initial_soc = (
                pybamm.AlgebraicSolver(tol=tol).solve(soc_model, [0])["soc"].data[0]
            )

            # Ensure that the result lies between 0 and 1
            initial_soc = np.minimum(np.maximum(initial_soc, 0.0), 1.0)

        elif isinstance(initial_value, (int, float)):
            if not 0 <= initial_value <= 1:
                raise ValueError("Initial SOC should be between 0 and 1")
            initial_soc = initial_value

        else:
            raise ValueError(
                "Initial value must be a float between 0 and 1, "
                "or a string ending in 'V'"
            )

        return initial_soc
