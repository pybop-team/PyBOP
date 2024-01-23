import pybop
import numpy as np
import warnings

## NOTE: This is a brittle example, the classes and methods below will be
## integrated into pybop in a future release.

# A design optimisation example loosely based on work by L.D. Couto
# available at https://doi.org/10.1016/j.energy.2022.125966.

# The target is to maximise the gravimetric energy density over a
# range of possible design parameter values, including for example:
# cross-sectional area = height x width (only need change one)
# electrode widths, particle radii, volume fractions and
# separator width.

# Define parameter set and model
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
model = pybop.lithium_ion.SPMe(parameter_set=parameter_set)

# Fitting parameters
parameters = [
    pybop.Parameter(
        "Positive electrode thickness [m]",
        prior=pybop.Gaussian(7.56e-05, 0.05e-05),
        bounds=[65e-06, 10e-05],
    ),
    pybop.Parameter(
        "Positive particle radius [m]",
        prior=pybop.Gaussian(5.22e-06, 0.05e-06),
        bounds=[2e-06, 9e-06],
    ),
]
# Define stoichiometries
sto = model._electrode_soh.get_min_max_stoichiometries(parameter_set)
mean_sto_neg = np.mean(sto[0:2])
mean_sto_pos = np.mean(sto[2:4])


# Define functions
def nominal_capacity(
    x, model
):  # > 50% of the simulation time is spent in this function (~0.7 sec per iteration vs ~1.1 for the forward simulation)
    """
    Update the nominal capacity based on the theoretical energy density and an
    average voltage.
    """
    inputs = {
        key: x[i] for i, key in enumerate([param.name for param in model.parameters])
    }
    model._parameter_set.update(inputs)

    theoretical_energy = model._electrode_soh.calculate_theoretical_energy(  # All of the computational time is in this line (~0.7)
        model._parameter_set
    )
    average_voltage = model._parameter_set["Positive electrode OCP [V]"](
        mean_sto_pos
    ) - model._parameter_set["Negative electrode OCP [V]"](mean_sto_neg)

    theoretical_capacity = theoretical_energy / average_voltage
    model._parameter_set.update({"Nominal cell capacity [A.h]": theoretical_capacity})


def cell_mass(parameter_set):  # This is very low compute time
    """
    Compute the total cell mass [kg] for the current parameter set.
    """

    # Approximations due to SPM(e) parameter set limitations
    electrolyte_density = parameter_set["Separator density [kg.m-3]"]

    # Electrode mass densities [kg.m-3]
    positive_mass_density = (
        parameter_set["Positive electrode active material volume fraction"]
        * parameter_set["Positive electrode density [kg.m-3]"]
    )
    +(parameter_set["Positive electrode porosity"] * electrolyte_density)
    negative_mass_density = (
        parameter_set["Negative electrode active material volume fraction"]
        * parameter_set["Negative electrode density [kg.m-3]"]
    )
    +(parameter_set["Negative electrode porosity"] * electrolyte_density)

    # Area densities [kg.m-2]
    positive_area_density = (
        parameter_set["Positive electrode thickness [m]"] * positive_mass_density
    )
    negative_area_density = (
        parameter_set["Negative electrode thickness [m]"] * negative_mass_density
    )
    separator_area_density = (
        parameter_set["Separator thickness [m]"]
        * parameter_set["Separator porosity"]
        * parameter_set["Separator density [kg.m-3]"]
    )
    positive_current_collector_area_density = (
        parameter_set["Positive current collector thickness [m]"]
        * parameter_set["Positive current collector density [kg.m-3]"]
    )
    negative_current_collector_area_density = (
        parameter_set["Negative current collector thickness [m]"]
        * parameter_set["Negative current collector density [kg.m-3]"]
    )

    # Cross-sectional area [m2]
    cross_sectional_area = (
        parameter_set["Electrode height [m]"] * parameter_set["Electrode width [m]"]
    )

    return cross_sectional_area * (
        positive_area_density
        + separator_area_density
        + negative_area_density
        + positive_current_collector_area_density
        + negative_current_collector_area_density
    )


# Define test protocol
experiment = pybop.Experiment(
    ["Discharge at 1C until 2.5 V (5 seconds period)"],
)
init_soc = 1  # start from full charge
signal = ["Voltage [V]", "Current [A]"]

# Generate problem
problem = pybop.DesignProblem(
    model, parameters, experiment, signal=signal, init_soc=init_soc
)

# Update the C-rate and the example dataset
nominal_capacity(problem.x0, model)
sol = problem.evaluate(problem.x0)
problem._time_data = sol[:, -1]
problem._target = sol[:, 0:-1]
dt = sol[1, -1] - sol[0, -1]


# Define cost function as a subclass
class GravimetricEnergyDensity(pybop.BaseCost):
    """
    Defines the (negative*) gravimetric energy density corresponding to a
    normalised 1C discharge from upper to lower voltage limits.
    *The energy density is maximised by minimising the negative energy density.
    """

    def __init__(self, problem):
        super().__init__(problem)

    def _evaluate(self, x, grad=None):
        """
        Compute the cost
        """
        with warnings.catch_warnings(record=True) as w:
            # Update the C-rate and run the simulation
            nominal_capacity(x, self.problem._model)
            sol = self.problem.evaluate(x)

            if any(w) and issubclass(w[-1].category, UserWarning):
                # Catch infeasible parameter combinations e.g. E_Li > Q_p
                print(f"ignoring this sample due to: {w[-1].message}")
                return np.inf

            else:
                voltage = sol[:, 0]
                current = sol[:, 1]
                gravimetric_energy_density = -np.trapz(
                    voltage * current, dx=dt
                ) / (  # trapz over-estimates compares to pybamm (~0.5%)
                    3600 * cell_mass(self.problem._model._parameter_set)
                )
            # Return the negative energy density, as the optimiser minimises
            # this function, to carry out maximisation of the energy density
            return gravimetric_energy_density


# Generate cost function and optimisation class
cost = GravimetricEnergyDensity(problem)
optim = pybop.Optimisation(
    cost, optimiser=pybop.PSO, verbose=True, allow_infeasible_solutions=False
)
optim.set_max_iterations(5)

# Run optimisation
x, final_cost = optim.run()
print("Estimated parameters:", x)
print(f"Initial gravimetric energy density: {-cost(cost.x0):.2f} Wh.kg-1")
print(f"Optimised gravimetric energy density: {-final_cost:.2f} Wh.kg-1")

# Plot the timeseries output
nominal_capacity(x, cost.problem._model)
pybop.quick_plot(x, cost, title="Optimised Comparison")

# Plot the cost landscape with optimisation path
if len(x) == 2:
    pybop.plot_cost2d(cost, optim=optim, steps=3)
