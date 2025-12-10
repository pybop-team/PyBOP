from copy import deepcopy
from multiprocessing import Pool

import matplotlib
import matplotlib.pyplot as plt
import pybamm
import sober
import torch
from ep_bolfi.models.solversetup import solver_setup, spectral_mesh_pts_and_method
from ep_bolfi.utility.preprocessing import calculate_desired_voltage
from numpy import arange, logspace, set_printoptions
from pandas import DataFrame
from scipy.stats import norm
from seaborn import kdeplot, pairplot
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sober import InverseModel

# torch.set_default_dtype(torch.float64)
# sober.setting_parameters(device=torch.device('cpu'))


def visualise_correlation(
    fig,
    ax,
    correlation,
    names=None,
    title=None,
    cmap=plt.get_cmap("BrBG"),
    entry_color="w",
):
    """
    Produces a heatmap of a correlation matrix.

    :param fig:
        The ``matplotlib.Figure`` object for plotting.
    :param ax:
        The ``matplotlib.Axes`` object for plotting.
    :param correlation:
        A two-dimensional (NumPy) array that is the correlation matrix.
    :param names:
        A list of strings that are names of the variables corresponding
        to each row or column in the correlation matrix.
    :param title:
        The title of the heatmap.
    :param cmap:
        The matplotlib colormap for the heatmap.
    :param entry_color:
        The colour of the correlation matrix entries.
    """

    # This one line produces the heatmap.
    ax.imshow(correlation, cmap=cmap, norm=matplotlib.colors.Normalize(-1, 1))
    # Define the coordinates of the ticks.
    ax.set_xticks(arange(len(correlation)))
    ax.set_yticks(arange(len(correlation)))
    # Display the names alongside the rows and columns.
    if names is not None:
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        # Rotate the labels at the x-axis for better readability.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Plot the correlation matrix entries on the heatmap.
    for i in range(len(correlation)):
        for j in range(len(correlation)):
            if i == j:
                color = "w"
            else:
                color = entry_color
            ax.text(
                j,
                i,
                f"{correlation[i][j]:3.2f}",
                ha="center",
                va="center",
                color=color,
                in_layout=False,
            )

    ax.set_title(title or "Correlation matrix")
    fig.colorbar(
        matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(-1, 1), cmap=cmap
        ),
        ax=ax,
        label="correlation",
    )
    fig.tight_layout()


torch.set_default_dtype(torch.float32)
sober.setting_parameters(dtype=torch.float32)  # , device=torch.device('cpu'))
torch.set_default_device(sober._settings._device)

seed = 0
model = pybamm.lithium_ion.DFN()
noise_generator = norm(0, 1e-4)


def simulator(parameters):
    global model, noise_generator

    tdf_curvature = parameters[0]
    D_e_scaling = parameters[1]
    D_e_magnitude = parameters[2]
    kappa_e_magnitude = parameters[3]
    # kappa_e_peak = parameters[4]
    # kappa_e_spread = parameters[5]
    kappa_e_peak = 1.0
    kappa_e_spread = 1.0

    def thermodynamic_factor(c_e, T):
        return 1 + tdf_curvature * (c_e / 1000) ** 2

    def electrolyte_diff(c_e, T):
        return D_e_magnitude * pybamm.exp(-D_e_scaling * (c_e / 1000 - 1))

    def electrolyte_cond(c_e, T):
        return (
            kappa_e_magnitude
            / (c_e / 1000)
            * pybamm.exp(
                -((pybamm.log(c_e / 1000 / kappa_e_peak) / kappa_e_spread) ** 2)
            )
        )

    model_parameters = pybamm.ParameterValues("Ramadass2004")
    model_parameters["Current function [A]"] = 5.0
    model_parameters["Thermodynamic factor"] = thermodynamic_factor
    model_parameters["Electrolyte diffusivity [m2.s-1]"] = electrolyte_diff
    model_parameters["Electrolyte conductivity [S.m-1]"] = electrolyte_cond
    discretization = {
        "order_s_n": 10,
        "order_s_p": 10,
        "order_e": 10,
        "volumes_e_n": 1,
        "volumes_e_s": 1,
        "volumes_e_p": 1,
        "halfcell": False,
    }
    solver = solver_setup(
        deepcopy(model),
        model_parameters,
        *spectral_mesh_pts_and_method(**discretization),
        verbose=False,
    )
    relaxation_t = logspace(-3, 2, 8)
    solution = solver(t_eval=relaxation_t)
    # relaxation_t = solution["Time [s]"].entries
    relaxation_U = calculate_desired_voltage(
        solution, relaxation_t, 1e-3, overpotential=True
    )
    relaxation_U += noise_generator.rvs(size=len(relaxation_U))

    return relaxation_t, relaxation_U, solution


def parallel_simulator(parameters):
    return simulator(parameters)[1]


def training_simulator(parameters):
    parameters = parameters.detach().cpu().numpy()
    with Pool() as p:
        results = p.map(parallel_simulator, parameters)
    return torch.tensor(results).to(dtype=torch.float32)


def clustering(data, max_number_of_clusters=10):
    clusterings = []
    labellings = []
    scores = []
    for i in range(2, max_number_of_clusters + 1):
        clusterings.append(KMeans(n_clusters=i))
        labellings.append(clusterings[-1].fit_predict(data))
        scores.append(silhouette_score(data, labellings[-1]))
    return clusterings, labellings, scores


if __name__ == "__main__":
    # Ramadass2004 values: D_e = 7.5e-10, kappa_e = 1.0, TDF = 1.0.
    # _, _, solution = simulator([1.0, 0.7, 7.5e-10, 1.0, 1.0, 1.0])
    bounds = torch.tensor(
        [
            [0.1, 0.2, 3e-10, 0.5],  # , 0.7, 0.7],
            [4.0, 2, 1e-9, 2.0],  # , 1.3, 1.5],
        ]
    )
    transforms = [(lambda x: torch.log(x), lambda x: torch.exp(x))] * 4  # 6
    names = [
        "TDF curv.",
        "Dₑ scaling",
        "Dₑ magn.",
        "κₑ magn.",
    ]  # "κₑ peak", "κₑ spread",

    inverse_modelling = InverseModel(
        training_simulator,
        model_initial_samples=2**5,
        bounds=bounds,
        prior="Uniform",
        transforms=transforms,
        seed=seed,
        disable_numpy_mode=True,
        parallelization=False,
        visualizations=False,
        names=names,
    )
    inverse_modelling.optimize_inverse_model_with_SOBER(
        stopping_criterion_variance=1e-8,
        maximum_number_of_batches=7,
        sober_iterations_per_convergence_check=7,
        sober_iterations_per_training_data_updates=1,
        model_samples_per_iteration=2**5,
        integration_nodes=2**5,
        visualizations=False,
        verbose=True,
    )

    relaxation_t, relaxation_U, solution = simulator(
        [1.0, 0.7, 7.5e-10, 1.0]  # , 1.0, 1.0]
    )
    features = training_simulator(
        torch.tensor([[1.0, 0.7, 7.5e-10, 1.0]])  # , 1.0, 1.0]])
    )

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
        s[0] for s in inverse_modelling.sample(features, 64).detach().cpu().numpy()
    ]
    with Pool() as p:
        simulations = p.map(simulator, samples)

    fig, ax = plt.subplots(figsize=(4 * 2**0.5, 4))
    ax.plot(relaxation_t / 3600, relaxation_U)
    """
    ax.fill_between(
        t_eval / 3600,
        sol_lower["Voltage [V]"](t_eval),
        sol_upper["Voltage [V]"](t_eval),
        alpha=0.3,
        color='grey'
    )
    """
    for sim in simulations:
        _, _, sol = sim
        ax.plot(
            sol["Time [h]"].entries,
            calculate_desired_voltage(
                sol, sol["Time [s]"].entries, 1e-3, overpotential=True
            ),
            alpha=0.5,
            lw=0.5,
            color="orange",
        )
    ax.set_xlabel("Time  /  h")
    ax.set_ylabel("Cell overpotential  /  mV")
    ax.set_xscale("log")

    # Get the whole multivariate normal distribution for correlations.
    prediction = inverse_modelling(features)
    covariance = prediction.covariance_matrix
    std = prediction.variance[0].sqrt()
    correlation = (covariance / std[:, None]) / std[None, :]
    fig_corr, ax_corr = plt.subplots(figsize=(3.75, 3))
    visualise_correlation(
        fig_corr,
        ax_corr,
        correlation.detach().cpu().numpy(),
        names,
        "Electrolyte characterization from pulse test",
        entry_color="black",
    )

    set_printoptions(precision=3)
    # Cluster the optimal evaulation points for analysis.
    normalized_X = inverse_modelling.tm.numpy(inverse_modelling.X_all)
    observations = inverse_modelling.tm.numpy(inverse_modelling.observations_all)
    clusterings, labellings, scores = clustering(normalized_X, max_number_of_clusters=4)
    for i, (clustering, labels, score) in enumerate(
        zip(clusterings, labellings, scores)
    ):
        print()
        print("Silhouette score of #" + str(i + 2) + " clusters:", score)
        labelled_X = (
            DataFrame(normalized_X)
            .set_axis(names, axis="columns")
            .assign(labels=labels)
        )
        correlation_plot = pairplot(labelled_X, height=1.25, hue="labels")
        correlation_plot.map_lower(kdeplot, levels=4)
        centers = clustering.cluster_centers_
        for j, center in enumerate(centers):
            # Find nearest observations.
            # index = argmin(sum((normalized_X - center)**2, axis=1))
            # X_index = normalized_X[index]
            # observation = observations[index]
            # Create new observations.
            features = training_simulator(torch.tensor(center).unsqueeze(0))
            prediction = inverse_modelling(features)
            model_center = (
                inverse_modelling.reverse_transform(
                    inverse_modelling.denormalize_input(
                        torch.tensor(
                            center, dtype=inverse_modelling.tm.dtype
                        ).unsqueeze(0)
                    )
                )[0]
                .detach()
                .cpu()
                .numpy()
            )
            model_prediction = (
                inverse_modelling.reverse_transform(
                    inverse_modelling.denormalize_input(prediction.mean)
                )[0]
                .detach()
                .cpu()
                .numpy()
            )
            print(
                "Parameters ",
                model_center,
                " - Prediction ",
                model_prediction,
                " = ",
                model_center - model_prediction,
            )
            covariance = prediction.covariance_matrix
            std = prediction.variance[0].sqrt()
            correlation = (covariance / std[:, None]) / std[None, :]
            fig_corr, ax_corr = plt.subplots(figsize=(3.75, 3))
            visualise_correlation(
                fig_corr,
                ax_corr,
                correlation.detach().cpu().numpy(),
                names,
                str(i + 2) + " clusters, label " + str(j) + ", " + str(center),
                entry_color="black",
            )
    plt.show()
