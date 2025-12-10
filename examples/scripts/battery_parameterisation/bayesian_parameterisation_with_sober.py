import importlib.util
import sys
import matplotlib
import matplotlib.pyplot as plt
from pyarrow import parquet
import numpy as np
from scipy.integrate import quad
from sober import setting_parameters
from torch import device, float64, set_default_dtype, tensor, zeros_like

from pybop.costs.feature_distances import indices_of
import pybop
from pybop.optimisers.sober_basq_optimiser import SOBER_BASQ_EPLFI_Options, SOBER_BASQ_EPLFI

from pybamm import citations, print_citations

set_default_dtype(float64)
setting_parameters(device=device('cpu'), dtype=float64)

seed = 0


def visualise_correlation(
    fig,
    ax,
    correlation,
    names=None,
    title=None,
    cmap=plt.get_cmap('BrBG'),
    entry_color='w'
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
    ax.set_xticks(np.arange(len(correlation)))
    ax.set_yticks(np.arange(len(correlation)))
    # Display the names alongside the rows and columns.
    if names is not None:
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        # Rotate the labels at the x-axis for better readability.
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Plot the correlation matrix entries on the heatmap.
    for i in range(len(correlation)):
        for j in range(len(correlation)):
            if i == j:
                color = 'w'
            else:
                color = entry_color
            ax.text(j, i, '{:3.2f}'.format(correlation[i][j]), ha='center',
                    va='center', color=color, in_layout=False)

    ax.set_title(title or "Correlation matrix")
    fig.colorbar(matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(-1, 1), cmap=cmap
    ), ax=ax, label="correlation")
    fig.tight_layout()


class Diffusive_Relaxation:
    """ Solution to ∂ₜ u = D ∂ₓ² u with u(x, t=0) = f(x). """

    def __init__(self, f, L, summands=10, radial=False):
        self.f = f
        self.L = L
        self.summands = summands
        self.radial = radial
        if radial:
            pass
        else:
            self.series = np.cos
            self.coefficients = self.compute_zero_flow_coefficients()

    def compute_zero_flow_coefficients(self):
        coefficients = tensor([
            2.0 / self.L * quad(
                lambda x: self.f(x) * np.cos(n * np.pi * x / self.L), 0, self.L
            )[0]
            for n in range(0, self.summands)
        ])
        # In order to use a simple summation expression suitable for
        # automatic differentiation, half the "zeroth" coefficient.
        coefficients[0] = coefficients[0] / 2.0
        return coefficients

    def concentration(self, x, t, D=1.0):
        value = zeros_like(D * t)
        for n in range(self.summands):
            value += (
                self.coefficients[n]
                * self.series(n * np.pi * x / self.L)
                * np.exp(
                    -n**2 * np.pi**2 * D * t / self.L**2
                )
            )
        return value

    def __call__(self, t, offset=0.0, timescale=1.0, magnitude=1.0):
        D = self.L**2 / timescale
        return offset + magnitude * (
            self.concentration(self.L, t, D) - self.concentration(self.L, t, D)
            - self.concentration(1.0, 0.0, D) + self.concentration(0.0, 0.0, D)
        )


class SiliconVoltageRelaxation(pybop.BaseSimulator):

    def __init__(self, parameters, fixed_parameters, timepoints: np.ndarray | None = None):
        citations.register("""@article{
            Köbbing2024,
            title={{Slow Voltage Relaxation of Silicon Nanoparticles with a Chemo-Mechanical Core-Shell Model}},
            author={Köbbing, L and Kuhn, Y and Horstmann, B},
            journal={ACS Applied Materials & Interfaces},
            volume={16},
            pages={67609-67619},
            year={2024}
        }""")
        super().__init__(parameters)
        self.fixed_parameters = fixed_parameters
        self.timepoints = np.asarray(timepoints)
        self.output_variables = ["Voltage change [V]"]

    def voltage_relaxation(self, inputs_array):

        # Parameters from Köbbing2024; notation is taken from there.
        R = 50e-9
        D = 1e-17
        E_core = 200e9
        nu_core = 0.22
        lambda_core = 64e9
        G_core = 82e9
        sigma_Y_core = 3e9
        c_max = 311e3
        SOC_0 = 0.1 * c_max
        SOC_100 = 0.9 * c_max
        v_Li = 9e-6
        L_shell = 20e-9
        E_shell = 100e9
        nu_shell = 0.3
        sigma_Y_shell = 2e9
        eta_shell = 135e12
        # sigma_ref = 133e6
        # tau = 3e8
        T = 298
        F = 96485
        R_gas = 8.314

        diffusion_model = Diffusive_Relaxation(lambda x: x, L=1)
        diffusion = False

        R_core = R - L_shell
        alpha = 0.5 * (R_core / L_shell - 1)

        U_infty = inputs_array[0].reshape(-1, 1)  # torch: unsqueeze(1)
        slope = inputs_array[1].reshape(-1, 1)
        timescale_exp = inputs_array[2].reshape(-1, 1)
        if diffusion:
            diff_portion = inputs_array[3].reshape(-1, 1)
            diff_timescale = inputs_array[4].reshape(-1, 1)
        else:
            diff_portion = 0
        rel_magnitude = (1 - diff_portion) * U_infty
        diff_magnitude = diff_portion * U_infty
        # Express effective parameters by adjusting model parameters.
        lambda_ch = 1  # does not appear independently
        sigma_ref = (
            slope * alpha * F * lambda_ch**3 / (2 * v_Li)
        )
        tau = lambda_ch * E_core * alpha * timescale_exp / sigma_ref
        # Determine integration constant from t=0 with
        # sigma_ev(t=0) = sigma_0 from sigma_ev = delta_U * F / v_Li.
        integration_constant = np.tanh(
            rel_magnitude * alpha * F * lambda_ch**3 / (2 * v_Li * sigma_ref)
        )
        # sigma_0 = np.arctanh(integration_constant) * (
        #     2 * sigma_ref / (alpha * lambda_ch**3)
        # )
        delta_U = 2 * v_Li * sigma_ref / (alpha * F * lambda_ch**3) * np.arctanh(
            integration_constant * np.exp(
                - E_core * alpha * lambda_ch / (
                    tau * sigma_ref
                ) * self.timepoints
            )
        )
        if diffusion:
            delta_U += diffusion_model(
                self.timepoints, -diff_magnitude, diff_timescale, diff_magnitude
            )

        return (U_infty - delta_U).T

    def batch_solve(self, inputs, calculate_sensitivities=False):
        inputs_array = tensor([entry for entry in inputs[0].values()])
        relaxations = self.voltage_relaxation(inputs_array)
        sols = []
        for entry, rel in zip(inputs, relaxations):
            sol = pybop.Solution(entry)
            sol.set_solution_variable("Voltage change [V]", rel)
            sols.append(sol)
        return sols


if __name__ == "__main__":
    data_index = 16

    spec = importlib.util.spec_from_file_location("read_dataset", "../../data/Wycisk2024/read_dataset.py")
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
            relaxations[i]["Voltage change [V]"][0] - u for u in relaxations[i]["Voltage change [V]"]
        ]
    # The first timepoint is set to 0, which messes up the plots.
    # The second timepoint is three orders of magnitude smaller than the
    # next, which messes up the parameterization.
    for i in range(len(relaxations)):
        relaxations[i]["Time [s]"] = relaxations[i]["Time [s]"][2:]
        relaxations[i]["Current function [A]"] = relaxations[i]["Current function [A]"][2:]
        relaxations[i]["Voltage change [V]"] = relaxations[i]["Voltage change [V]"][2:]

    t = np.asarray(relaxations[data_index]["Time [s]"])
    short_term_end = indices_of(t, 1e2)[0]
    long_term_start = indices_of(t, 1e4)[0]

    dataset = pybop.Dataset({
        "Time [s]": t,
        "Current function [A]": np.asarray(relaxations[data_index]["Current function [A]"]),
        "Voltage change [V]": np.asarray(relaxations[data_index]["Voltage change [V]"]),
    })
    len_data = len(t)

    short_term = pybop.MeanSquaredError(
        dataset,
        "Voltage change [V]",
        [1] * short_term_end + [0] * (len_data - short_term_end)
    )
    mid_term = pybop.MeanSquaredError(
        dataset,
        "Voltage change [V]",
        [0] * short_term_end + [1] * (long_term_start - short_term_end) + [0] * (len_data - long_term_start)
    )
    long_term = pybop.MeanSquaredError(
        dataset,
        "Voltage change [V]",
        [0] * long_term_start + [1] * (len_data - long_term_start)
    )

    unknowns = pybop.MultivariateParameters(
        {
            "U(t=∞) [V]": pybop.Parameter(
                initial_value=0.1,
                bounds=[0.01, 0.2],
                transformation=pybop.LogTransformation()
            ),
            "log-slope [V]": pybop.Parameter(
                initial_value=0.01,
                bounds=[0.001, 0.2],
                transformation=pybop.LogTransformation()
            ),
            "relaxation timescale [s]": pybop.Parameter(
                initial_value=1e5,
                bounds=[1e3, 1e7],
                transformation=pybop.LogTransformation()
            ),
        },
        distribution=pybop.MultivariateUniform(np.asarray([[0.01, 0.2], [0.001, 0.2], [1e3, 1e7]]))
    )
    simulator = SiliconVoltageRelaxation(unknowns, {}, timepoints=t)
    # Override the forced univariate Parameters
    simulator.parameters = unknowns
    problem = pybop.MetaProblem(pybop.Problem(simulator, mid_term), pybop.Problem(simulator, long_term))
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
    pybop_wrapper = SOBER_BASQ_EPLFI(problem, options)
    pybop_result = pybop_wrapper.run()
    # Calculate the correlation matrix in addition to the full plot.
    raw_taken_samples = pybop_result.posterior.distribution.distribution.dataset
    raw_mean = np.mean(raw_taken_samples, axis=1)
    raw_cov = np.cov(raw_taken_samples)
    raw_std = np.var(raw_taken_samples, axis=1)**0.5
    raw_corr = (raw_cov / raw_std[:, None]) / raw_std[None, :]
    fig_corr, ax_corr = plt.subplots(figsize=(3.75, 3))
    visualise_correlation(
        fig_corr, ax_corr, raw_corr, ["U(t=∞) [V]", "log-slope [V]", "relaxation timescale [s]"], "", entry_color='white'
    )
    # Re-sample the posterior for the predictive posterior.
    posterior_resamples = pybop_result.posterior.rvs(64, apply_transform=True)
    posterior_resamples_pdf = pybop_result.posterior.pdf(posterior_resamples)
    simulations = simulator.voltage_relaxation(posterior_resamples.T)
    fig_pos, ax_pos = plt.subplots(
        figsize=(3 * 2**0.5, 3), layout="constrained"
    )
    norm = matplotlib.colors.Normalize(
        posterior_resamples_pdf.min(), posterior_resamples_pdf.max()
    )
    cmap = plt.get_cmap('viridis')
    for pr, pr_pdf, u in zip(
        posterior_resamples.T,
        posterior_resamples_pdf,
        simulations.T
    ):
        ax_pos.semilogx(
            t,
            u,
            color=cmap(norm(pr_pdf)),
            lw=0.8,
            ls=':',
        )
    ax_pos.semilogx(
        tensor(t),
        tensor(relaxations[data_index]["Voltage change [V]"]),
        color='black',
        lw=2,
        label="experimental data"
    )
    fig_pos.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax_pos,
        label="Posterior PDF from KDE approximation"
    )
    ax_pos.set_xlabel("Time since start of relaxation  /  s")
    ax_pos.set_ylabel("Voltage change since start of relaxation  /  V")
    ax_pos.legend()

    print_citations()

    plt.show()
