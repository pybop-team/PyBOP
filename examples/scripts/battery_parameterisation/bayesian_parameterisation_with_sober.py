import importlib.util
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pybamm import print_citations
from sober import setting_parameters
from torch import device, float64, set_default_dtype, tensor

import pybop
from pybop.costs.feature_distances import indices_of
from pybop.models.lithium_ion.silicon_relaxation import SiliconRelaxation
from pybop.optimisers.sober_basq_optimiser import (
    SOBER_BASQ_EPLFI,
    SOBER_BASQ_EPLFI_Options,
)

set_default_dtype(float64)
setting_parameters(device=device("cpu"), dtype=float64)

seed = 0

if __name__ == "__main__":
    data_index = 16
    folder = os.path.dirname(os.path.realpath(__file__))

    spec = importlib.util.spec_from_file_location(
        "read_dataset",
        os.path.join(folder, "..", "..", "data", "Wycisk2024", "read_dataset.py"),
    )
    read_dataset = importlib.util.module_from_spec(spec)
    sys.modules["read_dataset"] = read_dataset
    spec.loader.exec_module(read_dataset)
    measurement = read_dataset.gitt_on_graphite_with_5_percent_silicon

    # pulses = measurement.get_subset(list(range(41, 82, 2)))
    relaxations = measurement.get_subset(list(range(42, 83, 2)))
    for i in range(len(relaxations)):
        relaxations[i]["Time [s]"] = [
            t - relaxations[i]["Time [s]"][0] for t in relaxations[i]["Time [s]"]
        ]
        relaxations[i]["Voltage change [V]"] = [
            relaxations[i]["Voltage change [V]"][0] - u
            for u in relaxations[i]["Voltage change [V]"]
        ]
    # The first timepoint is set to 0, which messes up the plots.
    # The second timepoint is three orders of magnitude smaller than the
    # next, which messes up the parameterization.
    for i in range(len(relaxations)):
        relaxations[i]["Time [s]"] = relaxations[i]["Time [s]"][2:]
        relaxations[i]["Current function [A]"] = relaxations[i]["Current function [A]"][
            2:
        ]
        relaxations[i]["Voltage change [V]"] = relaxations[i]["Voltage change [V]"][2:]

    t = np.asarray(relaxations[data_index]["Time [s]"])
    short_term_end = indices_of(t, 1e2)[0]
    long_term_start = indices_of(t, 1e4)[0]

    dataset = pybop.Dataset(
        {
            "Time [s]": t,
            "Current function [A]": np.asarray(
                relaxations[data_index]["Current function [A]"]
            ),
            "Voltage [V]": np.asarray(relaxations[data_index]["Voltage change [V]"]),
        }
    )
    len_data = len(t)

    short_term = pybop.MeanSquaredError(
        dataset,
        "Voltage [V]",
        [1] * short_term_end + [0] * (len_data - short_term_end),
    )
    mid_term = pybop.MeanSquaredError(
        dataset,
        "Voltage [V]",
        [0] * short_term_end
        + [1] * (long_term_start - short_term_end)
        + [0] * (len_data - long_term_start),
    )
    long_term = pybop.MeanSquaredError(
        dataset,
        "Voltage [V]",
        [0] * long_term_start + [1] * (len_data - long_term_start),
    )

    unknowns = pybop.MultivariateParameters(
        {
            "Terminal voltage of observed relaxation [V]": pybop.Parameter(
                initial_value=0.1,
                bounds=[0.01, 0.2],
                transformation=pybop.LogTransformation(),
            ),
            "Logarithmic slope of observed mechanical relaxation [V]": pybop.Parameter(
                initial_value=0.01,
                bounds=[0.001, 0.2],
                transformation=pybop.LogTransformation(),
            ),
            "Exponential timescale for decay of observed mechanical relaxation [s]": pybop.Parameter(
                initial_value=1e5,
                bounds=[1e3, 1e7],
                transformation=pybop.LogTransformation(),
            ),
        },
        distribution=pybop.MultivariateUniform(
            np.asarray([[0.01, 0.2], [0.001, 0.2], [1e3, 1e7]])
        ),
    )
    model = SiliconRelaxation()
    simulator = pybop.pybamm.simulator.Simulator(model, model.default_parameter_values)

    # Override the forced univariate Parameters
    simulator.parameters = unknowns
    problem = pybop.MetaProblem(
        pybop.Problem(simulator, mid_term), pybop.Problem(simulator, long_term)
    )

    # Copy the MultivariateParameters to the meta-problem
    problem.parameters = simulator.parameters
    options = SOBER_BASQ_EPLFI_Options(
        model_initial_samples=128,
        seed=seed,
        # disable_numpy_mode=True,
        # parallelisation=False,
        ep_iterations=2,
        ep_total_dampening=0.5,
        sober_iterations=4,
        model_samples_per_iteration=64,
        ep_integration_nodes=32,
        integration_nodes=512,
        batched_input=True,
    )
    optim = SOBER_BASQ_EPLFI(problem, options=options)
    result = optim.run()

    # Calculate the correlation matrix in addition to the full plot.
    raw_taken_samples = result.posterior.distribution.distribution.dataset
    raw_mean = np.mean(raw_taken_samples, axis=1)
    raw_cov = np.cov(raw_taken_samples)
    raw_std = np.var(raw_taken_samples, axis=1) ** 0.5
    raw_corr = (raw_cov / raw_std[:, None]) / raw_std[None, :]
    fig_corr, ax_corr = plt.subplots(figsize=(3.75, 3))
    pybop.plot.correlation(
        fig_corr,
        ax_corr,
        raw_corr,
        names=["U(t=∞) [V]", "log-slope [V]", "relaxation timescale [s]"],
        title="",
        entry_color="white",
    )

    # Re-sample the posterior for the predictive posterior.
    posterior_resamples = result.posterior.rvs(64, apply_transform=True)
    posterior_resamples_pdf = result.posterior.pdf(posterior_resamples)
    simulations = simulator.voltage_relaxation(posterior_resamples.T)
    fig_pos, ax_pos = plt.subplots(figsize=(3 * 2**0.5, 3), layout="constrained")
    norm = matplotlib.colors.Normalize(
        posterior_resamples_pdf.min(), posterior_resamples_pdf.max()
    )
    cmap = plt.get_cmap("viridis")
    for _pr, pr_pdf, u in zip(
        posterior_resamples.T, posterior_resamples_pdf, simulations.T, strict=False
    ):
        ax_pos.semilogx(
            t,
            u,
            color=cmap(norm(pr_pdf)),
            lw=0.8,
            ls=":",
        )
    ax_pos.semilogx(
        tensor(t),
        tensor(relaxations[data_index]["Voltage change [V]"]),
        color="black",
        lw=2,
        label="experimental data",
    )
    fig_pos.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax_pos,
        label="Posterior PDF from KDE approximation",
    )
    ax_pos.set_xlabel("Time since start of relaxation  /  s")
    ax_pos.set_ylabel("Voltage change since start of relaxation  /  V")
    ax_pos.legend()

    print_citations()

    plt.show()
