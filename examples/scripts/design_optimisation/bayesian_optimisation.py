# Miscellaneous imports for plotting, arithmetic, and statistics.

# In this example, we use parallel processing.

import matplotlib.pyplot as plt
import numpy as np
import pybamm
from matplotlib.ticker import PercentFormatter
from pybamm import print_citations
from sober import setting_parameters

# Imports the SOBER interface and ensures that calculations are on CPU.
from torch import device, float64, set_default_dtype

import pybop
from pybop.optimisers.sober_basq_optimiser import SOBER_BASQ, SOBER_BASQ_Options

# Sets the precision from 32-bit to 64-bit and forces CPU over GPU calculations.
set_default_dtype(float64)
setting_parameters(dtype=float64, device=device("cpu"))
np.seterr(divide="ignore")

"""
This example demonstrates how to call SOBER to perform a simple optimisation.
The example considers a heavily simplified operation cycle of a battery used
with a solar panel to provide energy at night. The optimisation task is to find
the right trade-off size between cost for additional capacity and the extended
life-time of the battery. Note: batteries degrade faster at extreme charge states.
"""


def day_night_cycle(oversize_factor):
    """
    Generates a cycling protocol based on the fraction of unused cell,
    and runs the SEI growth model with it. The cycling protocol emulates
    a solar panel-coupled battery over 10 years.

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
    currents = np.asarray(
        (
            [reference_current * 14 / 20] * 6
            + [0.0] * 2
            + [-reference_current * 6 / 20] * 14
            + [0.0] * 2
        )
        * 365
        * 10
    )  # roughly 10 years
    timepoints = np.asarray(range(len(currents))) * dt
    return timepoints, currents


def capacity_loss_cutoff(variables):
    return 0.6 - variables["Capacity lost to SEI [C]"] / (
        3600 * variables["Nominal cell capacity [A.h]"]
    )


def set_up_simulator(reference_lifetime=-1):
    """
    We need to set up the model twice. First we get the reference lifetime,
    then we use it to set up the target relative to that reference.
    """
    sei_model = pybamm.lithium_ion.SPM(
        options={"SEI": "VonKolzenberg2020", "surface form": "algebraic"}
    )
    # Add two variables to the model: one for cutting off at the EOL,
    # and one for describing the gain function to optimise.
    sei_model.variables["Capacity lost to SEI [C]"] = (
        (
            sei_model.variables["Negative SEI thickness [m]"]
            - battery_parameters["Initial SEI thickness [m]"]
        )
        * battery_parameters["Negative electrode surface area to volume ratio [m-1]"]
        * battery_parameters["Electrode width [m]"]
        * battery_parameters["Electrode height [m]"]
        * battery_parameters["Negative electrode thickness [m]"]
        / battery_parameters["SEI partial molar volume [m3.mol-1]"]
        * F
    )
    capacity_loss_termination = pybamm.step.CustomTermination(
        name="Capacity loss cut-off", event_function=capacity_loss_cutoff
    )
    if reference_lifetime > 0:
        sei_model.variables["Lifetime gained per oversize factor [d]"] = (
            sei_model.variables["Time [d]"] - reference_lifetime
        ) / (sei_model.parameters["Oversize factor"] - 1)
    protocol = pybamm.Experiment(
        [
            "Charge at 0.28 C for 6 hours",
            "Rest for 2 hours",
            "Discharge at 0.12 C for 14 hours",
            "Rest for 2 hours",
        ]
        * 3650,
        period="1 hour",
        termination=[capacity_loss_termination],
    )
    solar_battery_model = pybop.pybamm.Simulator(
        sei_model,
        battery_parameters,
        protocol=protocol,
        output_variables=["Voltage [V]"],
    )
    return solar_battery_model


if __name__ == "__main__":
    battery_parameters = pybamm.ParameterValues("Marquis2019")

    F = 96485.33212
    # We append parameters from a cell we know the SEI parameters for.
    battery_parameters.update(
        {
            "Negative electrode surface area to volume ratio [m-1]": 3
            * (1 - battery_parameters["Negative electrode porosity"])
            / battery_parameters["Negative particle radius [m]"],
            "Electrolyte diffusivity [m2.s-1]": 2.8e-10,
            "Electrode width [m]": (1 / 24) ** 0.5,
            "Nominal cell capacity [A.h]": 0.05,
            "SEI reaction exchange current density [A.m-2]": 1e-5,
            "SEI lithium ion conductivity [S.m-1]": 1e-8,
            "Initial concentration in negative electrode [mol.m-3]": (
                0.11
                * battery_parameters[
                    "Maximum concentration in negative electrode [mol.m-3]"
                ]
            ),
            "Initial concentration in positive electrode [mol.m-3]": (
                0.83
                * battery_parameters[
                    "Maximum concentration in positive electrode [mol.m-3]"
                ]
            ),
            # "SEI growth transfer coefficient": 0.22,  # currently not supported in PyBaMM
            "Inital SEI thickness [m]": 2e-9,
            "Tunneling distance for electrons [m]": 2.05e-9,  # tunneling within SEI
            "SEI partial molar volume [m3.mol-1]": 1.078e-5,
            "SEI lithium interstitial diffusivity [m2.s-1]": 1e-15,
            # "SEI relative permittivity": 131,  # not required for growth
            "SEI open-circuit potential [V]": 17400 / F,
            # "Anion transference number in SEI": 1 - 0.063,  # t_plus is 0.063, not required for growth
            "SEI porosity": 0.1,
            # "SEI Bruggeman coefficient": 4.54,  # not required for growth
            "Relative capacity cut-off for End-Of-Life": 0.4,
            # Change cell capacity by adjusting cross-section area.
            "Electrode height [m]": (1 / 24) ** 0.5
            * pybamm.Parameter("Oversize factor"),
            "Oversize factor": "[input]",
        },
        check_already_exists=False,
    )

    pybop_prior = pybop.MultivariateParameters(
        {"Adjustment factor": pybop.Parameter(initial_value=1.1, bounds=[1.0, 1.2])},
        distribution=pybop.MultivariateUniform(
            np.asarray([[1.0, 1.2 - 1.0]])
        ),  # bug work-around; change to actual [1.0, 1.2] bounds when PR862 is merged
    )
    # In this simple example, we first plot the whole target function.
    # A note, as this is a common point of confusion: this plot solves
    # the optimization already, as we can see the optimum point.
    # The point of this example is to show the application of the
    # optimiser in an easily verifiable example. As soon as many
    # variables are to be optimised at once, say 5 or more, or the
    # landscape of the target function becomes much more complex,
    # we use to optimiser to get away with much sparser evaluations.
    reference_lifetime = set_up_simulator().solve()["Time [d]"].entries[-1]
    simulator = set_up_simulator(reference_lifetime)
    cost = pybop.DesignCost(target="Time [d]")
    oversize_factors = np.asarray([1 + 0.002 * i for i in range(1, 101)])
    eol_days = [reference_lifetime]
    solar_battery_model = set_up_simulator(reference_lifetime)
    for oversize_factor in oversize_factors:
        eol_days.append(
            cost.evaluate(
                solar_battery_model.solve(inputs={"Oversize factor": oversize_factor})
            )
        )
    oversize_factors = np.append([1.0], oversize_factors)
    eol_days = np.asarray(eol_days)

    fig_kde, ax_kde = plt.subplots(figsize=(3 * 2**0.5, 3), layout="constrained")
    ax_eol = ax_kde.twinx()
    eol_plot = ax_eol.plot(
        oversize_factors, eol_days, label="Time to End-Of-Life  /  d"
    )[0]
    gain = [float("NaN")] + list(
        (eol_days[1:] - eol_days[0]) / (oversize_factors[1:] - 1)
    )
    gain_plot = ax_eol.plot(
        oversize_factors, gain, label="Gain per extra capacity  /  d"
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
    cost.target_data = np.asarray([0])
    pybop_problem = pybop.Problem(solar_battery_model, cost)
    pybop_problem.parameters = pybop_prior
    pybop_options = SOBER_BASQ_Options(
        model_initial_samples=16,
        maximise=True,
        sober_iterations=7,
        model_samples_per_iteration=16,
        integration_nodes=128,
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
        eval_kde, post_approx, label="Posterior for optimal sizing", ls="--"
    )[0]
    ax_kde.set_xlabel("Oversize factor")
    ax_kde.set_ylabel("Posterior probability density")
    ax_kde.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    plots_for_legend = [eol_plot, gain_plot, kde_plot]
    fig_kde.legend(
        plots_for_legend,
        [p.get_label() for p in plots_for_legend],
        loc="outside lower center",
        # borderpad=0.3,
        # handlelength=0.5,
        # handletextpad=0.3,
        # borderaxespad=0.0,
        # columnspacing=0.5,
    )

    print_citations()

    plt.show()
