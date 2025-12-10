from copy import deepcopy
from ep_bolfi.models.solversetup import (
    simulation_setup, spectral_mesh_pts_and_method
)
from ep_bolfi.utility.fitting_functions import fit_sqrt
from itertools import cycle
import matplotlib.pyplot as plt
from sober import InverseModel
from multiprocessing import Pool
import pybamm
from scipy.stats import norm
import sober
import torch

torch.set_default_dtype(torch.float64)
sober.setting_parameters(device=torch.device('cpu'))

seed = 0
model = pybamm.lithium_ion.DFN()
rest_duration = 300
rest_fraction_used = 0.1
period = 0.1
noise_generator = norm(0, 1e-4)


def simulator(parameters):
    global model, rest_duration, rest_fraction_used, period, noise_generator

    pulse_strength = parameters[0]
    pulse_length = parameters[1]
    model_parameters = model.default_parameter_values

    procedure = [
        "Discharge at "
        + str(pulse_strength)
        + " C for "
        + str(pulse_length)
        + " seconds ("
        + str(period)
        + " second period)",
        "Rest for " + str(rest_duration) + " seconds (1 second period)"
    ]
    discretization = {
        'order_s_n': 10, 'order_s_p': 10, 'order_e': 10,
        'volumes_e_n': 1, 'volumes_e_s': 1, 'volumes_e_p': 1,
        'halfcell': False
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
        pulse_end:pulse_end + int(rest_fraction_used * rest_duration)
    ]
    relaxation_U = solution["Voltage [V]"].entries[
        pulse_end:pulse_end + int(rest_fraction_used * rest_duration)
    ]
    relaxation_U += noise_generator.rvs(size=len(relaxation_U))

    return relaxation_t, relaxation_U, solution


def training_simulator(parameters):
    parameters = parameters.detach().cpu().numpy()
    relaxation_t, relaxation_U, _ = simulator(parameters)
    sqrt_features = fit_sqrt(relaxation_t, relaxation_U)[2]
    sqrt_features = torch.tensor(sqrt_features)
    sqrt_features[1] = torch.log(sqrt_features[1])
    return sqrt_features


if __name__ == "__main__":
    bounds = torch.tensor([
        [0.02, 10.0],
        [1.0, 600.0],
    ])
    transforms = [
        (lambda x: torch.log(x), lambda x: torch.exp(x)),
        (lambda x: torch.log(x), lambda x: torch.exp(x)),
    ]

    inverse_modelling = InverseModel(
        training_simulator,
        model_initial_samples=128,
        bounds=bounds,
        prior='Uniform',
        transforms=transforms,
        seed=seed,
        disable_numpy_mode=True,
        visualizations=False,
        names=["Pulse strength [C]", "Pulse length [s]"]
    )
    inverse_modelling.optimize_inverse_model_with_SOBER(
        stopping_criterion_variance=1e-12,
        maximum_number_of_batches=3,
        model_samples_per_iteration=128,
        integration_nodes=100,
        visualizations=False,
        verbose=True
    )

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

    samples = [
        s[0] for s in inverse_modelling.sample(features, 32).numpy()
    ]
    with Pool() as p:
        simulations = p.map(simulator, samples)

    fig, ax = plt.subplots(figsize=(3 * 2**0.5, 3))
    ax.plot(
        (relaxation_t - relaxation_t[0]) / 3600,
        relaxation_U,
        label="observed portion of the data"
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
    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()['color'])
    for simulation, sample, color in zip(simulations, samples, color_cycle):
        _, _, solution = simulation
        t = solution["Time [h]"].entries
        U = solution["Voltage [V]"].entries
        U += noise_generator.rvs(size=len(U))
        pulse_length = sample[1]
        pulse_end = int(pulse_length / period) + 1
        t_pulse = t[:pulse_end] - t[pulse_end]
        U_pulse = U[:pulse_end]
        t_rest = t[
            pulse_end + int(rest_fraction_used * rest_duration):
        ] - t[pulse_end]
        U_rest = U[pulse_end + int(rest_fraction_used * rest_duration):]
        ax.plot(t_pulse, U_pulse, alpha=0.5, lw=0.5, color=color)
        ax.plot(t_rest, U_rest, alpha=0.5, lw=0.5, color=color)
    ax.set_xlabel("Time  /  h")
    ax.set_ylabel("Cell voltage  /  V")
    ax.legend()
    plt.show()
