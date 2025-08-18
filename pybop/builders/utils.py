import numpy as np
from pybamm import LithiumIonParameters, Parameter, ParameterValues

import pybop


def create_weighting(weighting: str, dataset: pybop.Dataset, domain: str) -> np.ndarray:
    if weighting is None or weighting == "equal":
        return np.asarray(1.0)
    elif weighting == "domain":
        return _set_cost_domain_weighting(dataset, domain)
    else:
        raise ValueError(
            "cost.weighting must be 'equal', 'domain', or a custom numpy array"
            f", got {weighting}"
        )


def _set_cost_domain_weighting(dataset, domain) -> np.ndarray:
    """Calculate domain-based weighting."""
    domain_data = dataset[domain]
    domain_spacing = domain_data[1:] - domain_data[:-1]
    mean_spacing = np.mean(domain_spacing)

    # Create a domain weighting array in one operation
    return np.concatenate(
        (
            [(mean_spacing + domain_spacing[0]) / 2],
            (domain_spacing[1:] + domain_spacing[:-1]) / 2,
            [(domain_spacing[-1] + mean_spacing) / 2],
        )
    ) * ((len(domain_data) - 1) / (domain_data[-1] - domain_data[0]))


def set_formation_concentrations(parameter_values: ParameterValues) -> None:
    """
    Compute the concentration of lithium in the positive electrode assuming that
    all lithium in the active material originated from the positive electrode.

    Only perform the calculation if an initial concentration exists for both
    electrodes, i.e. it is not a half cell.

    Parameters
    ----------
    parameters : pybamm.ParameterValues
        A PyBaMM parameter set containing standard lithium-ion parameters.
    """
    params = parameter_values

    required_keys = {
        "Initial concentration in negative electrode [mol.m-3]",
        "Initial concentration in positive electrode [mol.m-3]",
    }

    if not required_keys.issubset(params.keys()):
        raise ValueError("Required keys not in ParameterValues object")

    # Obtain the total amount of lithium in the active material
    Q_Li_particles_init = params.evaluate(LithiumIonParameters().Q_Li_particles_init)

    volume_denominator = (
        params["Positive electrode active material volume fraction"]
        * params["Positive electrode thickness [m]"]
        * params["Electrode height [m]"]
        * params["Electrode width [m]"]
        * params["Faraday constant [C.mol-1]"]
    )

    # Convert the total amount to a concentration in the positive electrode
    c_init = Q_Li_particles_init * 3600 / volume_denominator

    params.update(
        {
            "Initial concentration in negative electrode [mol.m-3]": 0,
            "Initial concentration in positive electrode [mol.m-3]": c_init,
        }
    )


def cell_mass(parameter_values: ParameterValues) -> None:
    """
    Calculate the total cell mass in kilograms.

    This method uses the provided parameter set to calculate the mass of different
    components of the cell, such as electrodes, separator, and current collectors,
    based on their densities, porosities, and thicknesses. It then calculates the
    total mass by summing the mass of each component and adds it as a parameter,
    `Cell mass [kg]` in the parameter_values dictionary.

    Parameters
    ----------
    parameter_values : dict
        A dictionary containing the parameter values necessary for the calculation.

    """
    params = parameter_values

    # Pre-calculate cross-sectional area
    cross_sectional_area = Parameter("Electrode height [m]") * Parameter(
        "Electrode width [m]"
    )

    def electrode_mass_density(electrode_type):
        """Calculate mass density for positive or negative electrode."""
        prefix = f"{electrode_type} electrode"
        active_vol_frac = Parameter(f"{prefix} active material volume fraction")
        density = Parameter(f"{prefix} active material density [kg.m-3]")
        porosity = Parameter(f"{prefix} porosity")
        electrolyte_density = Parameter("Electrolyte density [kg.m-3]")
        cb_density = Parameter(f"{prefix} carbon-binder density [kg.m-3]")

        return (
            active_vol_frac * density
            + porosity * electrolyte_density
            + (1.0 - active_vol_frac - porosity) * cb_density
        )

    # Calculate all area densities
    area_densities = [
        # Electrodes
        Parameter("Positive electrode thickness [m]")
        * electrode_mass_density("Positive"),
        Parameter("Negative electrode thickness [m]")
        * electrode_mass_density("Negative"),
        # Separator
        Parameter("Separator thickness [m]") * Parameter("Separator density [kg.m-3]"),
        # Current collectors
        Parameter("Positive current collector thickness [m]")
        * Parameter("Positive current collector density [kg.m-3]"),
        Parameter("Negative current collector thickness [m]")
        * Parameter("Negative current collector density [kg.m-3]"),
    ]

    # Add cell mass to parameter_values
    params.update(
        {"Cell mass [kg]": cross_sectional_area * sum(area_densities)},
        check_already_exists=False,
    )
