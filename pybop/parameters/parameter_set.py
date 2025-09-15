from pybamm import (
    LithiumIonParameters,
)


def set_formation_concentrations(parameter_set):
    """
    Compute the concentration of lithium in the positive electrode assuming that
    all lithium in the active material originated from the positive electrode.

    Only perform the calculation if an initial concentration exists for both
    electrodes, i.e. it is not a half cell.

    Parameters
    ----------
    parameter_set : pybamm.ParameterValues
        A PyBaMM parameter set containing standard lithium ion parameters.
    """
    if (
        all(
            key in parameter_set.keys()
            for key in [
                "Initial concentration in negative electrode [mol.m-3]",
                "Initial concentration in positive electrode [mol.m-3]",
            ]
        )
        and parameter_set["Initial concentration in negative electrode [mol.m-3]"] > 0
    ):
        # Obtain the total amount of lithium in the active material
        Q_Li_particles_init = parameter_set.evaluate(
            LithiumIonParameters().Q_Li_particles_init
        )

        # Convert this total amount to a concentration in the positive electrode
        c_init = (
            Q_Li_particles_init
            * 3600
            / (
                parameter_set["Positive electrode active material volume fraction"]
                * parameter_set["Positive electrode thickness [m]"]
                * parameter_set["Electrode height [m]"]
                * parameter_set["Electrode width [m]"]
                * parameter_set["Faraday constant [C.mol-1]"]
            )
        )

        # Update the initial lithium concentrations
        parameter_set.update(
            {"Initial concentration in negative electrode [mol.m-3]": 0}
        )
        parameter_set.update(
            {"Initial concentration in positive electrode [mol.m-3]": c_init}
        )
