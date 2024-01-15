import pybamm
import warnings
from ..base_model import BaseModel


class SPM(BaseModel):
    """
    Wraps the Single Particle Model (SPM) for simulating lithium-ion batteries, as implemented in PyBaMM.

    The SPM is a simplified physics-based model that represents a lithium-ion cell using a single
    spherical particle to simulate the behavior of the negative and positive electrodes.

    Parameters
    ----------
    name : str, optional
        The name for the model instance, defaulting to "Single Particle Model".
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
    options : dict, optional
        A dictionary of options to customize the behavior of the PyBaMM model.
    """

    def __init__(
        self,
        name="Single Particle Model",
        parameter_set=None,
        geometry=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
        options=None,
    ):
        super().__init__()
        self.pybamm_model = pybamm.lithium_ion.SPM(options=options)
        self._unprocessed_model = self.pybamm_model
        self.name = name

        # Set parameters, using either the provided ones or the default
        self.default_parameter_values = self.pybamm_model.default_parameter_values
        self._parameter_set = (
            parameter_set or self.pybamm_model.default_parameter_values
        )
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

        self._electrode_soh = pybamm.lithium_ion.electrode_soh

    def _check_params(self, inputs=None, parameter_set=None, infeasible_locations=True):
        """
        A compatibility check for the model parameters which can be implemented by subclasses
        if required, otherwise it returns True by default.

        Parameters
        ----------
        inputs : dict
            The input parameters for the simulation.

        Returns
        -------
        bool
            A boolean which signifies whether the parameters are compatible.
        """
        parameter_set = parameter_set or self._parameter_set

        electrode_params = [
            (
                "Negative electrode active material volume fraction",
                "Negative electrode porosity",
            ),
            (
                "Positive electrode active material volume fraction",
                "Positive electrode porosity",
            ),
        ]

        related_parameters = {
            key: inputs.get(key) if inputs and key in inputs else parameter_set[key]
            for pair in electrode_params
            for key in pair
        }

        def warn_and_return(warn_message):
            warnings.warn(warn_message, UserWarning)
            return infeasible_locations

        for material_vol_fraction, porosity in electrode_params:
            if related_parameters[material_vol_fraction] + related_parameters[
                porosity
            ] > 1 and self.param_check_counter <= len(electrode_params):
                self.param_check_counter += 1
                return warn_and_return(
                    f"Non-physical point encountered - [{material_vol_fraction} + {porosity}] > 1.0!"
                )

        return True


class SPMe(BaseModel):
    """
    Represents the Single Particle Model with Electrolyte (SPMe) for lithium-ion batteries.

    The SPMe extends the basic Single Particle Model (SPM) by incorporating electrolyte dynamics,
    making it suitable for simulations where electrolyte effects are non-negligible. This class
    provides a framework to define the model parameters, geometry, mesh types, discretization
    points, spatial methods, and numerical solvers for simulation within the PyBaMM ecosystem.

    Parameters
    ----------
    name: str, optional
        A name for the model instance, defaults to "Single Particle Model with Electrolyte".
    parameter_set: pybamm.ParameterValues or dict, optional
        A dictionary or a ParameterValues object containing the parameters for the model. If None, the default PyBaMM parameters for SPMe are used.
    geometry: dict, optional
        A dictionary defining the model's geometry. If None, the default PyBaMM geometry for SPMe is used.
    submesh_types: dict, optional
        A dictionary defining the types of submeshes to use. If None, the default PyBaMM submesh types for SPMe are used.
    var_pts: dict, optional
        A dictionary specifying the number of points for each variable for discretization. If None, the default PyBaMM variable points for SPMe are used.
    spatial_methods: dict, optional
        A dictionary specifying the spatial methods for discretization. If None, the default PyBaMM spatial methods for SPMe are used.
    solver: pybamm.Solver, optional
        The solver to use for simulating the model. If None, the default PyBaMM solver for SPMe is used.
    options: dict, optional
        A dictionary of options to customize the behavior of the PyBaMM model.
    """

    def __init__(
        self,
        name="Single Particle Model with Electrolyte",
        parameter_set=None,
        geometry=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
        options=None,
    ):
        super().__init__()
        self.pybamm_model = pybamm.lithium_ion.SPMe(options=options)
        self._unprocessed_model = self.pybamm_model
        self.name = name

        # Set parameters, using either the provided ones or the default
        self.default_parameter_values = self.pybamm_model.default_parameter_values
        self._parameter_set = (
            parameter_set or self.pybamm_model.default_parameter_values
        )
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

        self._electrode_soh = pybamm.lithium_ion.electrode_soh

    def _check_params(self, inputs=None, parameter_set=None, infeasible_locations=True):
        """
        A compatibility check for the model parameters which can be implemented by subclasses
        if required, otherwise it returns True by default.

        Parameters
        ----------
        inputs : dict
            The input parameters for the simulation.

        Returns
        -------
        bool
            A boolean which signifies whether the parameters are compatible.
        """
        parameter_set = parameter_set or self._parameter_set

        electrode_params = [
            (
                "Negative electrode active material volume fraction",
                "Negative electrode porosity",
            ),
            (
                "Positive electrode active material volume fraction",
                "Positive electrode porosity",
            ),
        ]

        related_parameters = {
            key: inputs.get(key) if inputs and key in inputs else parameter_set[key]
            for pair in electrode_params
            for key in pair
        }

        def warn_and_return(warn_message):
            warnings.warn(warn_message, UserWarning)
            return infeasible_locations

        for material_vol_fraction, porosity in electrode_params:
            if related_parameters[material_vol_fraction] + related_parameters[
                porosity
            ] > 1 and self.param_check_counter <= len(electrode_params):
                self.param_check_counter += 1
                return warn_and_return(
                    f"Non-physical point encountered - [{material_vol_fraction} + {porosity}] > 1.0!"
                )

        return True
