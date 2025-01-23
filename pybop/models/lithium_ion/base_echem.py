import sys
import warnings
from typing import Optional

from pybamm import LithiumIonParameters, Simulation
from pybamm import lithium_ion as pybamm_lithium_ion

from pybop.models.base_model import BaseModel, Inputs
from pybop.parameters.parameter_set import ParameterSet


class EChemBaseModel(BaseModel):
    """
    Overwrites and extends `BaseModel` class for electrochemical PyBaMM models.

    Parameters
    ----------
    pybamm_model : pybamm.BaseModel
        A subclass of the pybamm Base Model.
    name : str, optional
        The name for the model instance, defaulting to "Electrochemical Base Model".
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
    eis : bool, optional
        A flag to build the forward model for EIS predictions. Defaults to False.
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
        name="Electrochemical Base Model",
        parameter_set=None,
        geometry=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
        eis=False,
        **model_kwargs,
    ):
        super().__init__(name=name, parameter_set=parameter_set, eis=eis)

        model_options = dict(build=False)
        for key, value in model_kwargs.items():
            model_options[key] = value
        self.pybamm_model = pybamm_model(**model_options)
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
        if solver is None:
            self._solver = self.pybamm_model.default_solver
            self._solver.mode = "fast with events"
            self._solver.max_step_decrease_count = 1
        else:
            self._solver = solver

        # Internal attributes for the built model are initialized but not set
        self._model_with_set_params = None
        self._built_model = None
        self._built_initial_soc = None
        self._mesh = None
        self._disc = None

        self._electrode_soh = pybamm_lithium_ion.electrode_soh
        self._electrode_soh_half_cell = pybamm_lithium_ion.electrode_soh_half_cell
        self.geometric_parameters = self.set_geometric_parameters()

    def _check_params(
        self,
        inputs: Inputs,
        parameter_set: ParameterSet,
        allow_infeasible_solutions: bool = True,
    ):
        """
        Check compatibility of the model parameters.

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
        if self.pybamm_model.options["working electrode"] == "positive":
            electrode_params = [
                (
                    "Positive electrode active material volume fraction",
                    "Positive electrode porosity",
                ),
            ]
        else:
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

        for material_vol_fraction, porosity in electrode_params:
            total_vol_fraction = (
                related_parameters[material_vol_fraction] + related_parameters[porosity]
            )
            if (
                ParameterSet.evaluate_symbol(total_vol_fraction, parameter_set)
                > 1 + sys.float_info.epsilon
            ):
                if self.param_check_counter <= len(electrode_params):
                    infeasibility_warning = "Non-physical point encountered - [{material_vol_fraction} + {porosity}] > 1.0!"
                    warnings.warn(infeasibility_warning, UserWarning, stacklevel=2)
                self.param_check_counter += 1
                return allow_infeasible_solutions

        return True

    def _set_initial_state(self, initial_state: dict, inputs: Optional[Inputs] = None):
        """
        Set the initial state of charge or concentrations for the battery model.

        Parameters
        ----------
        initial_state : dict
            A valid initial state, e.g. the initial state of charge or open-circuit voltage.
        inputs : Inputs
            The input parameters to be used when building the model.
        """
        initial_state = self.convert_to_pybamm_initial_state(initial_state)

        if not self.pybamm_model._built:  # noqa: SLF001
            self.pybamm_model.build_model()

        # Temporary construction of attributes for PyBaMM
        self._model = self.pybamm_model
        self._unprocessed_parameter_values = self._unprocessed_parameter_set

        # Set initial state via PyBaMM's Simulation class
        Simulation.set_initial_soc(self, initial_state, inputs=inputs)

        # Update the default parameter set for consistency
        self._unprocessed_parameter_set = self._parameter_values

        # Clear the pybamm objects
        del self._model
        del self._unprocessed_parameter_values
        del self._parameter_values

    def cell_volume(self, parameter_set: Optional[ParameterSet] = None):
        """
        Calculate the total cell volume in m3.

        This method uses the provided parameter set to calculate the total thickness
        of the cell including electrodes, separator, and current collectors. It then
        calculates the volume by multiplying by the cross-sectional area.

        Parameters
        ----------
        parameter_set : dict, optional
            A dictionary containing the parameter values necessary for the calculation.

        Returns
        -------
        float
            The total volume of the cell in m3.
        """
        parameter_set = parameter_set or self._parameter_set

        # Calculate cell thickness
        cell_thickness = (
            parameter_set["Positive electrode thickness [m]"]
            + parameter_set["Negative electrode thickness [m]"]
            + parameter_set["Separator thickness [m]"]
            + parameter_set["Positive current collector thickness [m]"]
            + parameter_set["Negative current collector thickness [m]"]
        )

        # Calculate cross-sectional area
        cross_sectional_area = (
            parameter_set["Electrode height [m]"] * parameter_set["Electrode width [m]"]
        )

        # Calculate total cell volume
        cell_volume = cross_sectional_area * cell_thickness

        return ParameterSet.evaluate_symbol(cell_volume, parameter_set)

    def cell_mass(self, parameter_set: Optional[ParameterSet] = None):
        """
        Calculate the total cell mass in kilograms.

        This method uses the provided parameter set to calculate the mass of different
        components of the cell, such as electrodes, separator, and current collectors,
        based on their densities, porosities, and thicknesses. It then calculates the
        total mass by summing the mass of each component.

        Parameters
        ----------
        parameter_set : dict, optional
            A dictionary containing the parameter values necessary for the calculation.

        Returns
        -------
        float
            The total mass of the cell in kilograms.
        """
        parameter_set = parameter_set or self._parameter_set

        def mass_density(
            active_material_vol_frac,
            density,
            porosity,
            electrolyte_density,
            carbon_binder_domain_density,
        ):
            return (
                (active_material_vol_frac * density)
                + (porosity * electrolyte_density)
                + (1.0 - active_material_vol_frac - porosity)
                * carbon_binder_domain_density
            )

        def area_density(thickness, mass_density):
            return thickness * mass_density

        # Calculate mass densities
        positive_mass_density = mass_density(
            parameter_set["Positive electrode active material volume fraction"],
            parameter_set["Positive electrode active material density [kg.m-3]"],
            parameter_set["Positive electrode porosity"],
            parameter_set["Electrolyte density [kg.m-3]"],
            parameter_set["Positive electrode carbon-binder density [kg.m-3]"],
        )
        negative_mass_density = mass_density(
            parameter_set["Negative electrode active material volume fraction"],
            parameter_set["Negative electrode active material density [kg.m-3]"],
            parameter_set["Negative electrode porosity"],
            parameter_set["Electrolyte density [kg.m-3]"],
            parameter_set["Negative electrode carbon-binder density [kg.m-3]"],
        )

        # Calculate area densities
        positive_area_density = area_density(
            parameter_set["Positive electrode thickness [m]"], positive_mass_density
        )
        negative_area_density = area_density(
            parameter_set["Negative electrode thickness [m]"], negative_mass_density
        )
        separator_area_density = area_density(
            parameter_set["Separator thickness [m]"],
            parameter_set["Separator density [kg.m-3]"],
        )
        positive_cc_area_density = area_density(
            parameter_set["Positive current collector thickness [m]"],
            parameter_set["Positive current collector density [kg.m-3]"],
        )
        negative_cc_area_density = area_density(
            parameter_set["Negative current collector thickness [m]"],
            parameter_set["Negative current collector density [kg.m-3]"],
        )

        # Calculate cross-sectional area
        cross_sectional_area = (
            parameter_set["Electrode height [m]"] * parameter_set["Electrode width [m]"]
        )

        # Calculate total cell mass
        total_area_density = (
            positive_area_density
            + negative_area_density
            + separator_area_density
            + positive_cc_area_density
            + negative_cc_area_density
        )
        cell_mass = cross_sectional_area * total_area_density

        return ParameterSet.evaluate_symbol(cell_mass, parameter_set)

    def approximate_capacity(self, parameter_set: Optional[ParameterSet] = None):
        """
        Calculate an estimate for the nominal cell capacity. The estimate is computed
        by estimating the capacity of the positive electrode that lies between the
        stoichiometric limits corresponding to the upper and lower voltage limits.

        Parameters
        ----------
        parameter_set : dict, optional
            A dictionary containing the parameter values necessary for the calculation.

        Returns
        -------
        float
            The estimate of the nominal cell capacity [A.h].
        """
        parameter_set = parameter_set or self._parameter_set

        # Calculate the theoretical capacity in the limit of low current
        if self.pybamm_model.options["working electrode"] == "positive":
            (
                max_sto_p,
                min_sto_p,
            ) = self._electrode_soh_half_cell.get_min_max_stoichiometries(parameter_set)
        else:
            (
                min_sto_n,
                max_sto_n,
                min_sto_p,
                max_sto_p,
            ) = self._electrode_soh.get_min_max_stoichiometries(parameter_set)
            # Note that the stoichiometric limits correspond to 0 and 100% SOC.
            # Stoichiometric balancing is performed within get_min_max_stoichiometries
            # such that the capacity accessible between the limits should be the same
            # for both electrodes, so we consider just the positive electrode below.

        Q_p = LithiumIonParameters().p.prim.Q_init
        theoretical_capacity = Q_p * (max_sto_p - min_sto_p)
        return ParameterSet.evaluate_symbol(theoretical_capacity, parameter_set)

    def set_geometric_parameters(self):
        """
        Sets the parameters that can be changed when rebuilding the model.

        Returns
        -------
        dict
            A dictionary of parameters that can be changed when rebuilding the model.

        """
        geometric_parameters = dict.fromkeys(
            [
                "Negative particle radius [m]",
                "Negative electrode porosity",
                "Negative electrode thickness [m]",
                "Positive particle radius [m]",
                "Positive electrode porosity",
                "Positive electrode thickness [m]",
                "Separator porosity",
                "Separator thickness [m]",
            ]
        )

        return geometric_parameters
