import pybamm


def add_variable_to_model(model, variable: str, target_time: float | None = None):
    """Supplement a lithium-ion model with a design variable."""

    if variable == "Gravimetric energy density [W.h.kg-1]":
        model.variables["Gravimetric energy density [W.h.kg-1]"] = (
            pybamm.ExplicitTimeIntegral(
                model.variables["Voltage [V]"]
                * model.variables["Current [A]"]
                / (3600 * pybamm.Parameter("Cell mass [kg]")),
                initial_condition=0.0,
            )
        )

    elif variable == "Volumetric energy density [W.h.m-3]":
        model.variables["Volumetric energy density [W.h.m-3]"] = (
            pybamm.ExplicitTimeIntegral(
                model.variables["Voltage [V]"]
                * model.variables["Current [A]"]
                / (3600 * pybamm.Parameter("Cell volume [m3]")),
                initial_condition=0.0,
            )
        )

    elif variable == "Gravimetric power density [W.kg-1]":
        model.variables["Gravimetric power density [W.kg-1]"] = (
            pybamm.ExplicitTimeIntegral(
                model.variables["Voltage [V]"]
                * model.variables["Current [A]"]
                / (target_time * pybamm.Parameter("Cell mass [kg]")),
                initial_condition=0.0,
            )
        )

    elif variable == "Volumetric power density [W.m-3]":
        model.variables["Volumetric power density [W.m-3]"] = (
            pybamm.ExplicitTimeIntegral(
                model.variables["Voltage [V]"]
                * model.variables["Current [A]"]
                / (target_time * pybamm.Parameter("Cell volume [m3]")),
                initial_condition=0.0,
            )
        )

    else:
        raise ValueError(f"Unrecognised variable name: {variable}")
