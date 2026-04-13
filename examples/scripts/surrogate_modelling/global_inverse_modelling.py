from copy import deepcopy
from itertools import cycle
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pybamm
import sober
import torch
from ep_bolfi.models.solversetup import simulation_setup, spectral_mesh_pts_and_method
from scipy.stats import norm

import pybop

torch.set_default_dtype(torch.float64)
sober.setting_parameters(device=torch.device("cpu"))

seed = 0
model = pybamm.lithium_ion.DFN()
rest_duration = 300
rest_fraction_used = 0.1
period = 0.1
noise_generator = norm(0, 1e-4)


class GITTSimulator(pybop.BaseSimulator):
    global model, rest_duration, rest_fraction_used, period, noise_generator

    def __init__(self, parameters):
        super().__init__(parameters)
        self.output_variables = ["Current [A]", "Voltage [V]"]

    def gitt_simulator(self, parameters):
        pulse_strength = parameters["Pulse strength [C]"]
        pulse_length = parameters["Pulse length [s]"]
        model_parameters = model.default_parameter_values

        procedure = [
            "Discharge at "
            + str(pulse_strength)
            + " C for "
            + str(pulse_length)
            + " seconds ("
            + str(period)
            + " second period)",
            "Rest for " + str(rest_duration) + " seconds (1 second period)",
        ]
        discretization = {
            "order_s_n": 10,
            "order_s_p": 10,
            "order_e": 10,
            "volumes_e_n": 1,
            "volumes_e_s": 1,
            "volumes_e_p": 1,
            "halfcell": False,
        }
        solver, _ = simulation_setup(
            deepcopy(model),
            procedure,
            model_parameters,
            *spectral_mesh_pts_and_method(**discretization),
            verbose=False,
        )
        solution = solver(calc_esoh=False)
        pulse_end = int(pulse_length / period) + 1
        relaxation_t = solution["Time [s]"].entries[
            pulse_end : pulse_end + int(rest_fraction_used * rest_duration)
        ]
        relaxation_U = solution["Voltage [V]"].entries[
            pulse_end : pulse_end + int(rest_fraction_used * rest_duration)
        ]
        relaxation_U += noise_generator.rvs(size=len(relaxation_U))

        solution["Time [s]"]._entries_raw -= pulse_length  # noqa: SL001

        return relaxation_t, relaxation_U, solution

    def solve_batch(self, inputs, calculate_sensitivities=False):
        sols = []
        for entry in inputs:
            _, _, solution = self.gitt_simulator(entry)
            sols.append(solution)
        return sols

    def __call__(self, parameters):
        return self.gitt_simulator(parameters)


if __name__ == "__main__":
    distribution = pybop.MultivariateUniform(np.asarray([[0.02, 1.0], [10.0, 600.0]]))
    pybop_prior = pybop.MultivariateParameters(
        {
            "Pulse strength [C]": pybop.Parameter(
                initial_value=0.2,
                bounds=[0.02, 1.0],
                transformation=pybop.LogTransformation(),
            ),
            "Pulse length [s]": pybop.Parameter(
                initial_value=90.0,
                bounds=[10.0, 600.0],
                transformation=pybop.LogTransformation(),
            ),
        },
        distribution=distribution,
    )
    # Overwrite the forced JointDistribution
    pybop_prior._distribution = distribution  # noqa: SL001
    simulator = GITTSimulator(pybop_prior)
    # Override the forced univariate Parameters
    simulator.parameters = pybop_prior
    # Use the "cost" functions as gain functions by setting the dataset to 0.
    GITT_cost_offset = pybop.SquareRootFeatureDistance(
        np.arange(900), np.zeros(900), feature="offset", time_start=0, time_end=30
    )
    GITT_cost_log_square_root = pybop.SquareRootFeatureDistance(
        np.arange(900), np.zeros(900), feature="log_slope", time_start=0, time_end=30
    )
    problem = pybop.MetaProblem(
        pybop.Problem(simulator, GITT_cost_offset),
        pybop.Problem(simulator, GITT_cost_log_square_root),
    )
    # Copy the MultivariateParameters to the meta-problem
    problem._parameters = simulator.parameters  # noqa: SL001

    options = pybop.SOBER_BASQ_GIS_Options(
        model_initial_samples=128,
        stopping_criterion_variance=1e-12,
        maximum_number_of_batches=3,
        model_samples_per_iteration=128,
        integration_nodes=100,
        verbose=True,
    )

    inverse_modelling = pybop.SOBER_BASQ_GIS(problem, options=options)

    relaxation_t, relaxation_U, solution = simulator([0.06, 80.0])
    features = training_simulator(torch.tensor([0.06, 80.0]))

    mean, _, (lower_bounds, upper_bounds) = inverse_modelling.evaluate(
        features, one_dimensional_confidence=True
    )

    print("Prediction:", mean)
    print("Lower bounds:", lower_bounds)
    print("Upper bounds:", upper_bounds)

    """
    _, _, sol_lower = simulator(lower_bounds[0].numpy())
    _, _, sol_upper = simulator(upper_bounds[0].numpy())
    t_eval = torch.linspace(
        0,
        min([
            sol_lower["Time [s]"].entries[-1],
            sol_upper["Time [s]"].entries[-1]
        ]),
        101
    ).numpy()
    """

    samples = [s[0] for s in inverse_modelling.sample(features, 32).numpy()]
    with Pool() as p:
        simulations = p.map(simulator, samples)

    fig, ax = plt.subplots(figsize=(3 * 2**0.5, 3))
    ax.plot(
        (relaxation_t - relaxation_t[0]) / 3600,
        relaxation_U,
        label="observed portion of the data",
    )
    """
    ax.fill_between(
        t_eval / 3600,
        sol_lower["Voltage [V]"](t_eval),
        sol_upper["Voltage [V]"](t_eval),
        alpha=0.3,
        color='grey'
    )
    """
    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    for simulation, sample, color in zip(
        simulations, samples, color_cycle, strict=False
    ):
        _, _, solution = simulation
        t = solution["Time [h]"].entries
        U = solution["Voltage [V]"].entries
        U += noise_generator.rvs(size=len(U))
        pulse_length = sample[1]
        pulse_end = int(pulse_length / period) + 1
        t_pulse = t[:pulse_end] - t[pulse_end]
        U_pulse = U[:pulse_end]
        t_rest = t[pulse_end + int(rest_fraction_used * rest_duration) :] - t[pulse_end]
        U_rest = U[pulse_end + int(rest_fraction_used * rest_duration) :]
        ax.plot(t_pulse, U_pulse, alpha=0.5, lw=0.5, color=color)
        ax.plot(t_rest, U_rest, alpha=0.5, lw=0.5, color=color)
    ax.set_xlabel("Time  /  h")
    ax.set_ylabel("Cell voltage  /  V")
    ax.legend()
    plt.show()
