import warnings

from ..base_model import BaseModel


class EChemBaseModel(BaseModel):
    """
    Overwrites and extends `BaseModel` class for electrochemical PyBaMM models.
    """

    def __init__(self):
        super().__init__()

    def _check_params(
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

        for material_vol_fraction, porosity in electrode_params:
            if (
                related_parameters[material_vol_fraction] + related_parameters[porosity]
                > 1
            ):
                if self.param_check_counter <= len(electrode_params):
                    infeasibility_warning = "Non-physical point encountered - [{material_vol_fraction} + {porosity}] > 1.0!"
                    warnings.warn(infeasibility_warning, UserWarning)
                self.param_check_counter += 1
                return allow_infeasible_solutions

        return True

    def cell_volume(self, parameter_set=None):
        """
        Calculate the total cell volume in m3.

        This method uses the provided parameter set to calculate the total thickness
        of the cell including electrodes, separator, and current collectors. It then
        calculates the volume by multiplying by the cross-sectional area.

        Parameters
        ----------
        parameter_set : dict, optional
            A dictionary containing the parameter values necessary for the volume
            calculation.

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

        # Calculate and return total cell volume
        return cross_sectional_area * cell_thickness

    def cell_mass(self, parameter_set=None):
        """
        Calculate the total cell mass in kilograms.

        This method uses the provided parameter set to calculate the mass of different
        components of the cell, such as electrodes, separator, and current collectors,
        based on their densities, porosities, and thicknesses. It then calculates the
        total mass by summing the mass of each component.

        Parameters
        ----------
        parameter_set : dict, optional
            A dictionary containing the parameter values necessary for the mass
            calculations.

        Returns
        -------
        float
            The total mass of the cell in kilograms.
        """
        parameter_set = parameter_set or self._parameter_set

        def mass_density(
            active_material_vol_frac, density, porosity, electrolyte_density
        ):
            return (active_material_vol_frac * density) + (
                porosity * electrolyte_density
            )

        def area_density(thickness, mass_density):
            return thickness * mass_density

        # Approximations due to SPM(e) parameter set limitations
        electrolyte_density = parameter_set["Separator density [kg.m-3]"]

        # Calculate mass densities
        positive_mass_density = mass_density(
            parameter_set["Positive electrode active material volume fraction"],
            parameter_set["Positive electrode density [kg.m-3]"],
            parameter_set["Positive electrode porosity"],
            electrolyte_density,
        )
        negative_mass_density = mass_density(
            parameter_set["Negative electrode active material volume fraction"],
            parameter_set["Negative electrode density [kg.m-3]"],
            parameter_set["Negative electrode porosity"],
            electrolyte_density,
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
            parameter_set["Separator porosity"] * electrolyte_density,
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

        # Calculate and return total cell mass
        total_area_density = (
            positive_area_density
            + negative_area_density
            + separator_area_density
            + positive_cc_area_density
            + negative_cc_area_density
        )
        return cross_sectional_area * total_area_density

    def approximate_capacity(self, x):
        """
        Calculate and update an estimate for the nominal cell capacity based on the theoretical
        energy density and an average voltage.

        The nominal capacity is computed by dividing the theoretical energy (in watt-hours) by
        the average open circuit potential (voltage) of the cell.

        Parameters
        ----------
        x : array-like
            An array of values representing the model inputs.

        Returns
        -------
        None
            The nominal cell capacity is updated directly in the model's parameter set.
        """
        # Extract stoichiometries and compute mean values
        (
            min_sto_neg,
            max_sto_neg,
            min_sto_pos,
            max_sto_pos,
        ) = self._electrode_soh.get_min_max_stoichiometries(self._parameter_set)
        mean_sto_neg = (min_sto_neg + max_sto_neg) / 2
        mean_sto_pos = (min_sto_pos + max_sto_pos) / 2

        inputs = {
            key: x[i] for i, key in enumerate([param.name for param in self.parameters])
        }
        self._parameter_set.update(inputs)

        # Calculate theoretical energy density
        theoretical_energy = self._electrode_soh.calculate_theoretical_energy(
            self._parameter_set
        )

        # Calculate average voltage
        positive_electrode_ocp = self._parameter_set["Positive electrode OCP [V]"]
        negative_electrode_ocp = self._parameter_set["Negative electrode OCP [V]"]
        try:
            average_voltage = positive_electrode_ocp(
                mean_sto_pos
            ) - negative_electrode_ocp(mean_sto_neg)
        except Exception as e:
            raise ValueError(f"Error in average voltage calculation: {e}")

        # Calculate and update nominal capacity
        theoretical_capacity = theoretical_energy / average_voltage
        self._parameter_set.update(
            {"Nominal cell capacity [A.h]": theoretical_capacity}
        )

    def set_rebuild_parameters(self):
        """
        Sets the parameters that can be changed when rebuilding the model.

        Returns
        -------
        dict
            A dictionary of parameters that can be changed when rebuilding the model.

        """
        rebuild_parameters = dict.fromkeys(
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

        return rebuild_parameters
