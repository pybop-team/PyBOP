from pybamm import LithiumIonParameters, Parameter, ParameterValues, Symbol


def set_formation_concentrations(parameter_values: ParameterValues) -> None:
    """
    Compute the concentration of lithium in the positive electrode assuming that
    all lithium in the active material originated from the positive electrode.

    Only perform the calculation if an initial concentration exists for both
    electrodes, i.e. it is not a half cell.

    Parameters
    ----------
    parameter_values : pybamm.ParameterValues
        A PyBaMM parameter set containing standard lithium ion parameters.
    """
    param_key = "Initial concentration in negative electrode [mol.m-3]"
    if param_key in parameter_values.keys() and parameter_values[param_key] > 0:
        # Obtain the total amount of lithium in the active material
        Q_Li_particles_init = parameter_values.evaluate(
            LithiumIonParameters().Q_Li_particles_init
        )

        # Convert this total amount to a concentration in the positive electrode
        c_init = parameter_values.evaluate(
            (Q_Li_particles_init * 3600)
            / (
                Parameter("Positive electrode active material volume fraction")
                * Parameter("Positive electrode thickness [m]")
                * electrode_area()
                * Parameter("Faraday constant [C.mol-1]")
            )
        )

        # Update the initial lithium concentrations
        parameter_values.update(
            {
                "Initial concentration in negative electrode [mol.m-3]": 0.0,
                "Initial concentration in positive electrode [mol.m-3]": c_init,
            }
        )


def electrode_area() -> Symbol:
    """
    An expression for the cross-sectional area of the electrode.
    """
    return Parameter("Electrode height [m]") * Parameter("Electrode width [m]")


def cell_volume() -> Symbol:
    """
    An expression for the total cell volume in m3.

    This method defines the total cell volume as the product of the electrode
    cross-sectional area and the total thickness of the cell including electrodes,
    eparator, and current collectors.
    """
    cell_thickness = (
        Parameter("Positive electrode thickness [m]")
        + Parameter("Negative electrode thickness [m]")
        + Parameter("Separator thickness [m]")
        + Parameter("Positive current collector thickness [m]")
        + Parameter("Negative current collector thickness [m]")
    )

    return electrode_area() * cell_thickness


def cell_mass() -> Symbol:
    """
    An expression for the total cell mass in kilograms.

    This method defines the total mass as the sum of the components masses,
    including the electrodes, separator, and current collectors, based on their
    densities, porosities, and thicknesses.
    """

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
            + (1.0 - active_material_vol_frac - porosity) * carbon_binder_domain_density
        )

    def area_density(thickness, mass_density):
        return thickness * mass_density

    # Mass densities
    positive_mass_density = mass_density(
        Parameter("Positive electrode active material volume fraction"),
        Parameter("Positive electrode active material density [kg.m-3]"),
        Parameter("Positive electrode porosity"),
        Parameter("Electrolyte density [kg.m-3]"),
        Parameter("Positive electrode carbon-binder density [kg.m-3]"),
    )
    negative_mass_density = mass_density(
        Parameter("Negative electrode active material volume fraction"),
        Parameter("Negative electrode active material density [kg.m-3]"),
        Parameter("Negative electrode porosity"),
        Parameter("Electrolyte density [kg.m-3]"),
        Parameter("Negative electrode carbon-binder density [kg.m-3]"),
    )

    # Areal densities
    positive_area_density = area_density(
        Parameter("Positive electrode thickness [m]"), positive_mass_density
    )
    negative_area_density = area_density(
        Parameter("Negative electrode thickness [m]"), negative_mass_density
    )
    separator_area_density = area_density(
        Parameter("Separator thickness [m]"),
        Parameter("Separator density [kg.m-3]"),
    )
    positive_cc_area_density = area_density(
        Parameter("Positive current collector thickness [m]"),
        Parameter("Positive current collector density [kg.m-3]"),
    )
    negative_cc_area_density = area_density(
        Parameter("Negative current collector thickness [m]"),
        Parameter("Negative current collector density [kg.m-3]"),
    )

    total_area_density = (
        positive_area_density
        + negative_area_density
        + separator_area_density
        + positive_cc_area_density
        + negative_cc_area_density
    )
    return electrode_area() * total_area_density
