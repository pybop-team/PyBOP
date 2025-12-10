# Miscellaneous imports for plotting, arithmetic, and statistics.
from copy import deepcopy
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

import pybop
from pybop.optimisers.sober_basq_optimiser import SOBER_BASQ_Options, SOBER_BASQ
import pybamm

# In this example, we use parallel processing.
from multiprocessing import Pool

# Imports the SOBER interface and ensures that calculations are on CPU.
from torch import device, float64, set_default_dtype
from sober import setting_parameters

from pybop.costs.feature_distances import indices_of
from pybop import BaseSimulator, Solution, Parameters

from pybamm import citations, print_citations

set_default_dtype(float64)
setting_parameters(dtype=float64, device=device('cpu'))
np.seterr(divide='ignore')


class SEIGrowthVonKolzenberg(BaseSimulator):
    """
    Implements the Solid-Electrolyte Interphase (SEI) growth model from
    von Kolzenberg et al. (2020).
    Extra model assumption: since the positive electrode is oversized,
    it can deliver extra lithium to the negative electrode.
    Currents are assumed to be slightly higher to account for this.
    """

    def __init__(self, parameters, fixed_parameters, timepoints: np.ndarray | None = None, currents: np.ndarray | None = None):
        citations.register("""@article{
            vonKolzenberg2020,
            title={{Solid-Electrolyte Interphase During Battery Cycling: Theory of Growth Regimes}},
            author={von Kolzenberg, L and Latz, A and Horstmann, B},
            journal={ChemSusChem},
            volume={13},
            pages={3901},
            year={2020},
            doi={10.1002/cssc.202000867}
        }""")
        super().__init__(parameters)
        self.fixed_parameters = fixed_parameters
        self.timepoints = timepoints
        self.currents = currents
        self.delta_t = np.diff(timepoints, append=timepoints[-1] - timepoints[-2])
        self.output_variables = ["SEI thickness [m]"]

    def sei_growth(self, inputs, timepoints=None, currents=None):

        timepoints = timepoints if timepoints is not None else self.timepoints
        currents = currents if currents is not None else self.currents
        delta_t = np.diff(timepoints, append=timepoints[-1] - timepoints[-2])

        local_parameters = deepcopy(self.fixed_parameters)
        local_parameters.update(inputs)

        negative_electrode_capacity = (
            (1 - local_parameters["Negative electrode porosity"])
            * local_parameters["Negative electrode thickness [m]"]
            * local_parameters["Electrode width [m]"]
            * local_parameters["Electrode height [m]"]
            * local_parameters["Maximum concentration in negative electrode [mol.m-3]"]
            * 96485.33212
        )
        SOC_start = (
            local_parameters["Initial concentration in negative electrode [mol.m-3]"]
            / local_parameters["Maximum concentration in negative electrode [mol.m-3]"]
        )
        socs = SOC_start + np.cumsum(currents) * delta_t / negative_electrode_capacity
        ocps = local_parameters["Negative electrode OCP [V]"](socs)

        R = 8.314462618  # Ideal gas constant
        T = local_parameters["Ambient temperature [K]"]
        F = 96485.33212  # Faraday's constant
        eff_surface_area = 3 * (1 - local_parameters["Negative electrode porosity"]) / local_parameters["Negative particle radius [m]"]
        # Assumption: exchange-current density is given as the pre-factor without concentration dependencies.
        exchange_current_density = local_parameters["Negative electrode exchange-current density [A.m-2]"](
            local_parameters["Initial concentration in electrolyte [mol.m-3]"],
            0.5,
            local_parameters["Maximum concentration in negative electrode [mol.m-3]"],
            local_parameters["Ambient temperature [K]"]
        ).value
        # Approximation: electrolyte concentration remains constant.
        exchange_current_densities = exchange_current_density * (1 - socs)**0.5 * socs**0.5

        # Approximation: the current sinked in the SEI is much smaller than the intercalation current.
        intercalation_currents = currents / (eff_surface_area * local_parameters["Negative electrode thickness [m]"])
        intercalation_overpotential = R * T / F * 2 * np.arcsinh(0.5 * intercalation_currents / exchange_current_densities)

        diffusion_critical_thicknesses = (
            local_parameters["Initial concentration in electrolyte [mol.m-3]"]
            * local_parameters["SEI diffusivity [m2.s-1]"]
            * 96485.33212
            / local_parameters["SEI formation rate constant [A.m-2]"]
            * np.exp(
                -(1 - local_parameters["SEI formation symmetry factor"])
                * F / (R * T) * (
                    intercalation_overpotential
                    + ocps
                    + local_parameters["SEI lithium reference potential [J.mol-1]"] / F
                )
            )
        )
        migration_critical_thicknesses = (
            2 * R * T * local_parameters["SEI ionic conductivity [S.m-1]"]
            / (F * np.abs(intercalation_currents))
        )

        sei_thicknesses = [local_parameters["Inital SEI thickness [m]"]] + [0.0 for _ in range(len(socs))]
        for i, dt in enumerate(delta_t):
            current_sei_thickness = sei_thicknesses[i]
            # This is basically SEI thickness minus tunneling effects,
            # but adjusted to allow for SEIs thinner than the tunneling length.
            apparent_sei_thickness = (
                (current_sei_thickness - local_parameters["SEI tunneling length [m]"]) / 2
                + (
                    ((current_sei_thickness - local_parameters["SEI tunneling length [m]"]) / 2)**10
                    + local_parameters["Reference apparent SEI thickness [m]"]**10
                )**0.1
            )
            sei_current = (
                -local_parameters["SEI formation rate constant [A.m-2]"]
                * np.exp(
                    -local_parameters["SEI formation symmetry factor"]
                    * F / (R * T) * (
                        intercalation_overpotential[i]
                        + ocps[i]
                        + local_parameters["SEI lithium reference potential [J.mol-1]"] / F
                    )
                ) * (
                    1 + apparent_sei_thickness / migration_critical_thicknesses[i]
                ) / (
                    1 + apparent_sei_thickness / migration_critical_thicknesses[i]
                    + apparent_sei_thickness / diffusion_critical_thicknesses[i]
                )
            )
            sei_thicknesses[i + 1] = (
                current_sei_thickness
                - local_parameters["SEI molar volume [m3.mol-1]"] / F * sei_current * dt
            )

        return np.asarray(sei_thicknesses[:-1])

    def batch_solve(self, inputs, calculate_sensitivities=False):
        sols = []
        for entry in inputs:
            sol = Solution(entry)
            sol.set_solution_variable("SEI thickness [m]", self.sei_growth(entry))
            sols.append(sol)
        return sols


class IdealisedSolarBatteryDegradation(BaseSimulator):
    """
    Generates a cycling protocol based on the fraction of unused cell,
    and runs the SEI growth model with it. The cycling protocol emulates
    a solar panel-coupled battery over 10 years.
    """

    def __init__(self, parameters, fixed_parameters, capacity_cutoff=0.4):
        super().__init__(parameters)
        self.fixed_parameters = fixed_parameters
        self.capacity_cutoff = capacity_cutoff
        # Calculate the reference case of a minimally sized battery.
        timepoints, currents = self.day_night_cycle(1.0)
        self.sei_growth_model = SEIGrowthVonKolzenberg(Parameters(), self.fixed_parameters, timepoints, currents)
        self.eol_reference = self.eol(1.0)
        self.output_variables = ["EOL [d]", "SEI thickness [m]"]

    def day_night_cycle(self, oversize_factor):
        """
        :param oversize_factor:
            A number greater than 1, giving the ratio of cell capacity to
            minimally required cell capacity.
        :returns:
            The SEI thickness over time.
        """
        reference_current = (1.0 / 2.5) / oversize_factor
        # With 1 hour per timestep, each cycle lasts 24 hours: discharge from
        # 17:00 to 7:00, rest until 9:00, charge until 15:00, and rest until 17:00.
        dt = 3600
        currents = np.asarray((
            [reference_current * 14 / 20] * 6 + [0.0] * 2 + [-reference_current * 6 / 20] * 14 + [0.0] * 2
        ) * 365 * 10)  # roughly 10 years
        timepoints = np.asarray(range(len(currents))) * dt
        return timepoints, currents

    def sei_growth(self, oversize_factor, return_plot_data=False):
        timepoints, currents = self.day_night_cycle(oversize_factor)
        sei_thicknesses = self.sei_growth_model.sei_growth({}, timepoints, currents)
        if return_plot_data:
            return sei_thicknesses, timepoints
        return sei_thicknesses

    def eol_formula(self, sei_thicknesses):
        relative_capacity_loss = (
            sei_thicknesses - sei_thicknesses[0]
        ) * (
            3 * (1 - self.fixed_parameters["Negative electrode porosity"]) / self.fixed_parameters["Negative particle radius [m]"]
        ) * 96485.33212 / (3600 * self.fixed_parameters["Nominal cell capacity [A.h]"])
        eol_day = indices_of(relative_capacity_loss, self.capacity_cutoff)[0] / 24
        return eol_day

    def eol(self, oversize_factor):
        sei_thicknesses = self.sei_growth(oversize_factor)
        return self.eol_formula(sei_thicknesses)

    def relative_eol_gain_formula(self, sei_thicknesses, oversize_factor):
        gain_vs_offset = (self.eol_formula(sei_thicknesses) - self.eol_reference) / (oversize_factor - 1)
        return gain_vs_offset

    def relative_eol_gain(self, oversize_factor):
        sei_thicknesses = self.sei_growth(oversize_factor)
        return self.relative_eol_gain_formula(sei_thicknesses, oversize_factor)

    def batch_solve(self, inputs, calculate_sensitivities=False):
        sols = []
        for entry in inputs:
            sol = Solution(entry)
            sei_thicknesses = self.sei_growth(entry["Oversize factor"])
            sol.set_solution_variable("SEI thickness [m]", sei_thicknesses)
            sol.set_solution_variable("EOL [d]", [self.relative_eol_gain_formula(sei_thicknesses, entry["Oversize factor"])])
            sols.append(sol)
        return sols


if __name__ == "__main__":

    battery_parameters = pybamm.ParameterValues("Marquis2019")

    # We append parameters from a cell we know the SEI parameters for.
    battery_parameters.update({
        "Electrolyte diffusivity [m2.s-1]": 2.8e-10,
        "Electrode width [m]": (1 / 24) ** 0.5,
        "Electrode height [m]": (1 / 24) ** 0.5,
        "Nominal cell capacity [A.h]": 0.05,
        "SEI formation rate constant [A.m-2]": 1e-5,
        "SEI ionic conductivity [S.m-1]": 1e-8,
        "Initial concentration in negative electrode [mol.m-3]": (
            0.11 * battery_parameters["Maximum concentration in negative electrode [mol.m-3]"]
        ),
        "Initial concentration in positive electrode [mol.m-3]": (
            0.83 * battery_parameters["Maximum concentration in positive electrode [mol.m-3]"]
        ),
        "SEI formation symmetry factor": 0.22,
        "Inital SEI thickness [m]": 2e-9,
        "Reference apparent SEI thickness [m]": 0.05e-9,
        "SEI tunneling length [m]": 2.05e-9,
        "SEI molar volume [m3.mol-1]": 1.078e-5,
        "SEI diffusivity [m2.s-1]": 1e-15,
        "SEI thickness [m]": 67e-9,
        "SEI relative permittivity": 131,
        "SEI lithium reference potential [J.mol-1]": 17400,
        "Anion transference number in SEI": 1 - 0.063,  # t_plus is 0.063
        "SEI porosity": 0.1,
        "SEI Bruggeman coefficient": 4.54,
    }, check_already_exists=False)

    pybop_prior = pybop.MultivariateParameters(
        {"Oversize factor": pybop.Parameter(initial_value=1.1, bounds=[1.0, 1.2])},
        distribution=pybop.MultivariateUniform(np.asarray([[1.0, 1.2]]))
    )
    # In this simple example, we first plot the whole target function.
    # A note, as this is a common point of confusion: this plot solves
    # the optimization already, as we can see the optimum point.
    # The point of this example is to show the application of the
    # optimiser in an easily verifiable example. As soon as many
    # variables are to be optimised at once, say 5 or more, or the
    # landscape of the target function becomes much more complex,
    # such a plot can not be reasonably produced, but the optimiser
    # still works just as well.
    solar_battery_model = IdealisedSolarBatteryDegradation(pybop_prior, battery_parameters)
    oversize_factors = np.asarray([1 + 0.002 * i for i in range(101)])
    with Pool() as p:
        eol_days = p.map(solar_battery_model.eol, oversize_factors)
    oversize_factors = oversize_factors.flatten()
    # End-Of-Life is chosen to be at 40% of capacity lost.
    fig_kde, ax_kde = plt.subplots(figsize=(3 * 2**0.5, 3), layout='constrained')
    ax_eol = ax_kde.twinx()
    eol_plot = ax_eol.plot(
        oversize_factors,
        eol_days,
        label="Time to End-Of-Life  /  d"
    )[0]
    gain = [float('NaN')] + list(
        (eol_days[1:] - eol_days[0]) / (oversize_factors[1:] - 1)
    )
    gain_plot = ax_eol.plot(
        oversize_factors,
        gain,
        label="Gain per extra capacity  /  d"
    )[0]
    ax_kde.set_xlabel("Oversize factor")
    ax_eol.set_ylabel("Time  /  d")

    # We have seen that the optimum ratio of battery lifetime gained to
    # battery oversizing is achieved at ~ 10.3% oversizing fraction.
    # Now we showcase how to obtain this result with SOBER instead.
    # We utilise the same interface as for the parameterization, and for
    # most of its arguments we refer you to its documentation.
    # In the special case of optimization, we set the 'data' argument to
    # a suitably-shaped 0,such that "target function - data" is just
    # the target function, and set 'maximize' to True.
    # Since it is needed for EOL gain calculation, we also pass the
    # reference EOL, which will be passed on to the target function.
    """
    sober_wrapper = SoberWrapper(
        calculate_eol_gain,
        tensor([0]),
        model_initial_samples=16,
        bounds=tensor([[1.0], [1.2]]),
        prior='Uniform',
        maximize=True,
        seed=0,
        names=["Oversize factor"],
        true_optimum=tensor([1.103]),
        offset=eol_indices[0] / 24
    )
    """

    # We now invoke SOBER to explore the target function efficiently.
    # Its settings, for which we refer you to its documentation, are
    # best found by trial-and-error and a coarse initial guess about
    # the complexity of the target function. Its results are stored
    # in the interface instance, which we will access later.
    """
    sober_wrapper.run_SOBER(
        sober_iterations=7,
        model_samples_per_iteration=16,
        visualizations=False,
        verbose=True
    )
    """

    # We now invoke BASQ to assess the quality with which SOBER has
    # explored the target function. Its settings, for which we refer
    # you to its documentation, are best found by trial-and-error and
    # a coarse initial guess about the complexity of the target
    # function. We get five return values:
    #  1. samples from the probability distribution that SOBER generated
    #     as a (faster) surrogate to the original target function,
    #  2. the optimal point in terms of the maximum value of the
    #     surrogate, called the Maximum A Posteriori (MAP) point,
    #  3. the optimal point that has been evaluated on the original
    #     target function,
    #  4. the SOBER approximation quality criterion, the expected log
    #     marginal likelihood (lower is better, scale is relative),
    #  5. the SOBER approximation quality criterion quality criterion,
    #     i.e., the self-assessment about the accuracy with which the
    #     expected log marginal likelihood was calculated, expressed
    #     in terms of the variance of the log marginal likelihood.
    """
    (
        taken_samples,
        MAP,
        best_observed,
        log_expected_marginal_likelihood,
        log_approx_variance_marginal_likelihood
    ) = sober_wrapper.run_BASQ(
        integration_nodes=128,
        visualizations=False,
        verbose=True
    )
    """

    cost = pybop.DesignCost("EOL [d]")
    cost._target_data = np.asarray([0])
    pybop_problem = pybop.Problem(solar_battery_model, cost)
    pybop_problem.parameters = pybop_prior
    pybop_options = SOBER_BASQ_Options(
        model_initial_samples=16,
        maximise=True,
        sober_iterations=7,
        model_samples_per_iteration=16,
        integration_nodes=128
    )
    sober_basq_wrapper = SOBER_BASQ(pybop_problem, pybop_options)
    pybop_result = sober_basq_wrapper.run()

    # sober_wrapper = sober_basq_wrapper.optim

    # We have seen visualizations form 'run_SOBER' and 'run_BASQ' about
    # their internal states. For a more intuitive visualization, we
    # employ the so-called predictive posterior. Rather than showing
    # the probability distribution of the model parameter values, we
    # plot the model realizations for a representative sample of it.
    # We will use the samples SOBER took and the samples BASQ generated
    # for plotting a so-called Kernel Density Estimate (KDE). If you are
    # not familiar with this, think of it as a spruced up histogram.
    eval_kde = np.linspace(1.0, 1.2, 101)
    post_approx = pybop_result.posterior.pdf(eval_kde)
    post_norm = sum(post_approx) * (1.2 - 1.0)
    post_approx /= post_norm
    kde_plot = ax_kde.plot(
        eval_kde, post_approx, label="Posterior for optimal sizing", ls='--'
    )[0]
    ax_kde.set_xlabel("Oversize factor")
    ax_kde.set_ylabel("Posterior probability density")
    ax_kde.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    plots_for_legend = [eol_plot, gain_plot, kde_plot]
    fig_kde.legend(
        plots_for_legend,
        [p.get_label() for p in plots_for_legend],
        loc='outside lower center',
        # borderpad=0.3,
        # handlelength=0.5,
        # handletextpad=0.3,
        # borderaxespad=0.0,
        # columnspacing=0.5,
    )

    print_citations()

    plt.show()
