from pybop.models.base_model import BaseModel


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
        **model_kwargs,
    ):
        model_options = dict(build=False)
        for key, value in model_kwargs.items():
            model_options[key] = value
        self.pybamm_model = pybamm_model(**model_options)
        self._unprocessed_model = self.pybamm_model

        # Correct OCP if set to default
        if (
            parameter_set is not None
            and "Open-circuit voltage [V]" in parameter_set.params
        ):
            default_ocp = self.pybamm_model.default_parameter_values[
                "Open-circuit voltage [V]"
            ]
            if parameter_set.params["Open-circuit voltage [V]"] == "default":
                print("Setting open-circuit voltage to default function")
                parameter_set.params["Open-circuit voltage [V]"] = default_ocp

        super().__init__(name=name, parameter_set=parameter_set)

        # Set parameters, using either the provided ones or the default
        self.default_parameter_values = self.pybamm_model.default_parameter_values
        self._parameter_set = self._parameter_set or self.default_parameter_values
        self._unprocessed_parameter_set = self._parameter_set

        # Define model geometry and discretization
        self.geometry = geometry or self.pybamm_model.default_geometry
        self.submesh_types = submesh_types or self.pybamm_model.default_submesh_types
        self.var_pts = var_pts or self.pybamm_model.default_var_pts
        self.spatial_methods = (
            spatial_methods or self.pybamm_model.default_spatial_methods
        )
        self.solver = solver or self.pybamm_model.default_solver

        # Internal attributes for the built model are initialized but not set
        self._model_with_set_params = None
        self._built_model = None
        self._built_initial_soc = None
        self._mesh = None
        self._disc = None
        self.geometric_parameters = {}

    def _check_params(self, inputs=None, allow_infeasible_solutions=True):
        """
        Check the compatibility of the model parameters.

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
