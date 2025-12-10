import importlib.util
import json
import sys
from contextlib import redirect_stdout
from multiprocessing import Pool

import matplotlib.pyplot as plt
import pybamm
import pybammeis
from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance
from ep_bolfi.models.solversetup import spectral_mesh_pts_and_method
from ep_bolfi.utility.dataset_formatting import read_parquet_table
from ep_bolfi.utility.fitting_functions import fit_drt
from ep_bolfi.utility.preprocessing import find_occurrences
from ep_bolfi.utility.visualization import (
    interactive_impedance_model,
    nyquist_plot,
)
from numpy import sqrt, tan
from scipy import constants
from scipy.stats import gaussian_kde
from torch import pi, tensor, zeros

spec_gunther = importlib.util.spec_from_file_location(
    "gunther", "../../data/Gunther2025/parameters.py"
)
gunther = importlib.util.module_from_spec(spec_gunther)
sys.modules["gunther"] = gunther
spec_gunther.loader.exec_module(gunther)
spec_kuhn = importlib.util.spec_from_file_location(
    "kuhn", "../../data/Kuhn2026/parameters.py"
)
kuhn = importlib.util.module_from_spec(spec_kuhn)
sys.modules["kuhn"] = kuhn
spec_kuhn.loader.exec_module(kuhn)

import numpy as np
from sober import SoberWrapper
from torch import exp, log

parameter_ranges = {
    "pouch_SPM": {
        "Positive particle diffusivity [m2.s-1]": (1e-16, 1e-14),
        "Positive electrode double-layer capacity [F.m-2]": (0.1, 0.5),
        "Positive electrode exchange-current density [A.m-2]": (0.1, 1.0),
    },
    "pouch_DFN": {
        "Positive electrode Bruggeman coefficient": (1, 2.5),
        "Positive electrode conductivity [S.m-1]": (0.01, 10),
        "Electrolyte diffusivity [m2.s-1]": (1e-10, 1e-9),
        "Positive particle diffusivity [m2.s-1]": (1e-16, 1e-14),
        "Positive electrode double-layer capacity [F.m-2]": (0.1, 0.5),
        "Positive electrode exchange-current density [A.m-2]": (0.1, 1.0),
    },
    "LG_MJ1_variant_1": {
        "Positive electrode double-layer capacity [F.m-2]": (7e-4, 7e-3),
        "Positive electrode exchange-current density [A.m-2]": (2e-2, 8e-2),
        "SEI relative permittivity": (100.0, 200.0),
        "SEI Bruggeman coefficient": (4.3, 4.8),
    },
    "LG_MJ1_variant_2": {
        "Positive electrode double-layer capacity [F.m-2]": (1e-2, 1e-1),
        "Positive electrode exchange-current density [A.m-2]": (2e-2, 8e-2),
        "SEI relative permittivity": (10.0, 50.0),
        "SEI Bruggeman coefficient": (4.3, 4.8),
    },
}

parameter_transforms = {
    "pouch_SPM": {
        "Positive particle diffusivity [m2.s-1]": (lambda x: log(x), lambda x: exp(x)),
        "Positive electrode double-layer capacity [F.m-2]": (
            lambda x: log(x),
            lambda x: exp(x),
        ),
        "Positive electrode exchange-current density [A.m-2]": (
            lambda x: log(x),
            lambda x: exp(x),
        ),
    },
    "pouch_DFN": {
        "Positive electrode Bruggeman coefficient": (lambda x: x, lambda x: x),
        "Positive electrode conductivity [S.m-1]": (lambda x: log(x), lambda x: exp(x)),
        "Electrolyte diffusivity [m2.s-1]": (lambda x: log(x), lambda x: exp(x)),
        "Positive particle diffusivity [m2.s-1]": (lambda x: log(x), lambda x: exp(x)),
        "Positive electrode double-layer capacity [F.m-2]": (
            lambda x: log(x),
            lambda x: exp(x),
        ),
        "Positive electrode exchange-current density [A.m-2]": (
            lambda x: log(x),
            lambda x: exp(x),
        ),
    },
    "LG_MJ1_variant_1": {
        "Positive electrode double-layer capacity [F.m-2]": (
            lambda x: log(x),
            lambda x: exp(x),
        ),
        "Positive electrode exchange-current density [A.m-2]": (
            lambda x: log(x),
            lambda x: exp(x),
        ),
        "SEI relative permittivity": (lambda x: log(x), lambda x: exp(x)),
        "SEI Bruggeman coefficient": (lambda x: x, lambda x: x),
    },
    "LG_MJ1_variant_2": {
        "Positive electrode double-layer capacity [F.m-2]": (
            lambda x: log(x),
            lambda x: exp(x),
        ),
        "Positive electrode exchange-current density [A.m-2]": (
            lambda x: log(x),
            lambda x: exp(x),
        ),
        "SEI relative permittivity": (lambda x: log(x), lambda x: exp(x)),
        "SEI Bruggeman coefficient": (lambda x: x, lambda x: x),
    },
}
parameter_transforms_numpy = {
    "pouch_SPM": {
        "Positive particle diffusivity [m2.s-1]": (
            lambda x: np.log(x),
            lambda x: np.exp(x),
        ),
        "Positive electrode double-layer capacity [F.m-2]": (
            lambda x: np.log(x),
            lambda x: np.exp(x),
        ),
        "Positive electrode exchange-current density [A.m-2]": (
            lambda x: np.log(x),
            lambda x: np.exp(x),
        ),
    },
    "pouch_DFN": {
        "Positive electrode Bruggeman coefficient": (lambda x: x, lambda x: x),
        "Positive electrode conductivity [S.m-1]": (
            lambda x: np.log(x),
            lambda x: np.exp(x),
        ),
        "Electrolyte diffusivity [m2.s-1]": (lambda x: np.log(x), lambda x: np.exp(x)),
        "Positive particle diffusivity [m2.s-1]": (
            lambda x: np.log(x),
            lambda x: np.exp(x),
        ),
        "Positive electrode double-layer capacity [F.m-2]": (
            lambda x: np.log(x),
            lambda x: np.exp(x),
        ),
        "Positive electrode exchange-current density [A.m-2]": (
            lambda x: np.log(x),
            lambda x: np.exp(x),
        ),
    },
    "LG_MJ1_variant_1": {
        "Positive electrode double-layer capacity [F.m-2]": (
            lambda x: np.log(x),
            lambda x: np.exp(x),
        ),
        "Positive electrode exchange-current density [A.m-2]": (
            lambda x: np.log(x),
            lambda x: np.exp(x),
        ),
        "SEI relative permittivity": (lambda x: np.log(x), lambda x: np.exp(x)),
        "SEI Bruggeman coefficient": (lambda x: x, lambda x: x),
    },
    "LG_MJ1_variant_2": {
        "Positive electrode double-layer capacity [F.m-2]": (
            lambda x: np.log(x),
            lambda x: np.exp(x),
        ),
        "Positive electrode exchange-current density [A.m-2]": (
            lambda x: np.log(x),
            lambda x: np.exp(x),
        ),
        "SEI relative permittivity": (lambda x: np.log(x), lambda x: np.exp(x)),
        "SEI Bruggeman coefficient": (lambda x: x, lambda x: x),
    },
}


def Z_SEI(f, parameters):
    """
    Impedance of the Solid-Electrolyte Interphase (Single2019).

    :param f:
        An array of the frequencies to evaluate.
    :param parameters:
        A dictionary of the model parameters.
    :returns:
        The evaluated impedances as an array.
    """

    εₙ = parameters["Negative electrode porosity"]
    βₙ = parameters["Negative electrode Bruggeman coefficient (electrolyte)"]
    Dₑ = parameters["Electrolyte diffusivity [m2.s-1]"]
    ε_SEI = parameters["SEI porosity"]
    β_SEI = parameters["SEI Bruggeman coefficient"]
    nₚ = parameters["Stoichiometry of cation in electrolyte salt dissociation"]
    nₙ = parameters["Stoichiometry of anion in electrolyte salt dissociation"]
    zₚ_salt = parameters["Charge number of cation in electrolyte salt dissociation"]
    zₙ_salt = parameters["Charge number of anion in electrolyte salt dissociation"]
    ρₑ = parameters["Mass density of electrolyte [kg.m-3]"]
    ρₑ_plus = parameters["Mass density of cations in electrolyte [kg.m-3]"]
    M_N = parameters["Molar mass of electrolyte solvent [kg.mol-1]"]
    c_N = parameters["Solvent concentration [mol.m-3]"]
    ρ_N = M_N * c_N
    v_N = parameters["Partial molar volume of electrolyte solvent [m3.mol-1]"]
    tilde_ρ_N = M_N / v_N
    one_plus_dlnf_dlnc = parameters["Thermodynamic factor"]
    Lₙ = parameters["Negative electrode thickness [m]"]
    Lₛ = parameters["Separator thickness [m]"]
    Lₚ = parameters["Positive electrode thickness [m]"]
    L_electrolyte_for_SEI_model = (((Lₙ + Lₚ) / 2) + Lₛ) / 2
    L_SEI = parameters["SEI thickness [m]"]
    t_SEI_minus = parameters["Anion transference number in SEI"]
    permittivity_SEI = parameters["SEI relative permittivity"]
    F = constants.physical_constants["Faraday constant"][0]
    ɛ_0 = constants.physical_constants["vacuum electric permittivity"][0]
    C_SEI = ɛ_0 * permittivity_SEI / L_SEI
    κ_SEI = parameters["SEI ionic conductivity [S.m-1]"]
    R_SEI = L_SEI / (ε_SEI**β_SEI * κ_SEI)

    # Notations refer to Single2019.
    k_electrolyte = (1 - 1j) * sqrt(εₙ ** (-βₙ) * 2 * pi * f / (2 * Dₑ))
    k_SEI = (1 - 1j) * sqrt(ε_SEI ** (1 - β_SEI) * 2 * pi * f / (2 * Dₑ))
    Theta = (
        -(nₚ + nₙ)
        / (nₚ * nₙ)
        / (zₚ_salt * zₙ_salt * F**2)
        * ρₑ**2
        / (ρ_N * tilde_ρ_N)
        * one_plus_dlnf_dlnc
    )
    Psi = 1 - ε_SEI ** ((1 + β_SEI) / 2) * tan(
        k_electrolyte * L_electrolyte_for_SEI_model
    ) * tan(k_SEI * L_SEI)
    Z_D_SEI = (
        L_SEI
        * Theta
        / (Dₑ * ε_SEI**β_SEI)
        * (t_SEI_minus - ρₑ_plus / ρₑ) ** 2
        * tan(k_SEI * L_SEI)
        / (Psi * k_SEI * L_SEI)
    )

    return 1 / (2 * pi * 1j * f * C_SEI + 1 / (R_SEI + Z_D_SEI))


def preprocess_data(data_index, electrode, cell_name):
    with open("../../data/Gunther2025/impedance_ocv_alignment.json") as f:
        ocv_alignment = json.load(f)[cell_name]
    raw_data_index = ocv_alignment["indices"].index(data_index)
    soc = ocv_alignment[electrode.capitalize() + " electrode SOC [-]"][raw_data_index]
    if cell_name == "18650_LG_3500_MJ1_EIS_anode_discharge":
        directory = "../../Kuhn2026/"
    else:
        directory = "../../Gunther2025/"
    data = read_parquet_table(directory + cell_name + ".parquet", "impedance")

    frequencies = tensor(data.frequencies[raw_data_index])
    impedances = tensor(data.complex_impedances[raw_data_index])

    return frequencies, impedances, soc


##########################################################
# Simulator used for visualization and parameterization. #
##########################################################


def composed_model(
    parameters,
    angular_frequencies,
    working_electrode="both",
    electrolyte=True,
    sei=False,
    three_electrode=None,
    reference_electrode_location=0.5,
):
    # "surface form": "differential" enables surface capacitance.
    model_options = {
        "surface form": "differential",
        "working electrode": working_electrode,
    }
    if electrolyte:
        model = pybamm.lithium_ion.DFN(options=model_options)
    else:
        model = pybamm.lithium_ion.SPM(options=model_options)
    discretization = {
        "order_s_n": 10,
        "order_s_p": 10,
        "order_e": 10,
        "volumes_e_n": 1,
        "volumes_e_s": 1,
        "volumes_e_p": 1,
        "halfcell": False,
    }
    eis_sim = pybammeis.EISSimulation(
        model,
        pybamm.ParameterValues(dict(parameters)),
        None,  # geometry
        *spectral_mesh_pts_and_method(**discretization),
        three_electrode=three_electrode,
        reference_electrode_location=reference_electrode_location,
    )
    impedance = eis_sim.solve(angular_frequencies)
    # Put the SEI analytic model in series to DFN with capacitance.
    if sei:
        impedance += Z_SEI(angular_frequencies, parameters)
    return impedance


def simulator(
    trial,
    angular_frequencies,
    cell_name,
    soc,
    electrolyte=True,
    variable_parameters=[],
):
    if cell_name == "18650_LG_3500_MJ1_EIS_anode_discharge":
        parameters = kuhn.get_parameters_with_switched_electrodes()
        sei = True
        working_electrode = "both"
        three_electrode = "positive"
        reference_electrode_location = 0.5
    else:
        parameters = gunther.get_parameters_for_cell(cell_name)
        sei = False
        working_electrode = "positive"
        three_electrode = "positive"
        reference_electrode_location = 1.0

    parameters["Positive electrode SOC"] = soc
    for i, key in enumerate(variable_parameters):
        parameters[key] = trial[i].item()

    return composed_model(
        parameters,
        angular_frequencies.detach().numpy(),
        working_electrode,
        electrolyte,
        sei,
        three_electrode,
        reference_electrode_location,
    )


def drt_simulator(
    trial,
    angular_frequencies,
    cell_name,
    soc,
    lambda_value=-2.0,
    num_data_peaks=None,
    electrolyte=True,
    variable_parameters=[],
):
    parallel_arguments = [
        (t, angular_frequencies, cell_name, soc, electrolyte, variable_parameters)
        for t in trial
    ]
    with Pool() as p:
        impedances = p.starmap(simulator, parallel_arguments)
    parallel_arguments = [(angular_frequencies, i, lambda_value) for i in impedances]
    with Pool() as p:
        drt_results = p.starmap(fit_drt, parallel_arguments)
    results = []
    for drt_tau, drt_resistance, _ in drt_results:
        num_data_peaks = num_data_peaks or len(drt_tau)
        results.append(
            tensor(
                [[[dr, dt]] * num_data_peaks for dr, dt in zip(drt_tau, drt_resistance)]
            ).log()
        )
    return results


def manual_model_assessment(
    frequencies,
    impedances,
    parameters,
    unknowns,
    transform_unknowns,
    working_electrode,
    electrolyte,
    sei,
    three_electrode,
    reference_electrode_location,
    lambda_value,
):
    interactive_impedance_model(
        frequencies.detach().numpy(),
        impedances.detach().numpy(),
        parameters,
        unknowns=unknowns,
        transform_unknowns=transform_unknowns,
        model=lambda par, freq: composed_model(
            par,
            freq / 1j,
            working_electrode,
            electrolyte,
            sei,
            three_electrode,
            reference_electrode_location,
        ),
        lambda_value=lambda_value,
    )
    plt.show()


def automated_model_assessment(
    frequencies,
    impedances,
    mean,
    bounds,
    transforms,
    cell_name,
    soc,
    electrolyte,
    variable_parameters,
    lambda_value,
):
    drt_tau, drt_resistance, drt = fit_drt(
        frequencies, impedances, lambda_value or -2.0
    )
    drt_data = tensor(list(zip(drt_tau, drt_resistance))).log()
    num_data_peaks = len(drt_tau)
    sober_wrapper = SoberWrapper(
        model=drt_simulator,
        data=drt_data,
        model_initial_samples=48,
        mean=mean,
        bounds=bounds,
        prior="Uniform",
        use_bolfi=False,
        transforms=transforms,
        disable_numpy_mode=True,
        parallelization=False,
        visualizations=False,
        names=variable_parameters,
        angular_frequencies=frequencies,
        cell_name=cell_name,
        soc=soc,
        lambda_value=drt.lambda_value,
        num_data_peaks=num_data_peaks,
        electrolyte=electrolyte,
        variable_parameters=variable_parameters,
    )

    # Re-define the distance function to counter the effect of less
    # simulator DRT peaks automatically lessening the error.
    # These are two versions to choose from. Taking the distance to all
    # data peaks is more stable, but has weak optima. Taking only the
    # distance to the nearest data peak leads to better convergence.

    def distance_function_all(observations):
        # Distance of every simulator peak to every data peak.
        num_sim_peaks = observations.shape[1]
        return ((observations - sober_wrapper.data) * sober_wrapper.weights).view(
            observations.shape[0], -1
        ).norm(dim=1).to(
            device=sober_wrapper.tm.device, dtype=sober_wrapper.tm.dtype
        ) / (num_sim_peaks / num_data_peaks) ** 0.5

    def distance_function_nearest(observations):
        # Distance of every simulator peak to its nearest data peak.
        num_sim_peaks = observations.shape[1]
        peak_distances = (
            (observations - sober_wrapper.data) * sober_wrapper.weights
        ).norm(dim=3) / (num_sim_peaks / num_data_peaks) ** 0.5
        distances = zeros(observations.shape[0]).to(
            device=sober_wrapper.tm.device, dtype=sober_wrapper.tm.dtype
        )
        for i in range(len(distances)):
            pd_entry = peak_distances[i].tolist()
            for _ in range(num_sim_peaks):
                nearest = tensor(pd_entry).argmin()
                distances[i] += pd_entry[nearest // num_data_peaks][
                    nearest % num_data_peaks
                ]
                pd_entry.pop(nearest // num_data_peaks)
        return distances

    sober_wrapper.distance_function = distance_function_nearest

    sober_wrapper.run_SOBER(
        sober_iterations=1,
        model_samples_per_iteration=48,
        acquisition_function=None,
        visualizations=False,
        verbose=True,
    )

    for _ in range(18):
        raw_taken_samples = sober_wrapper.run_BASQ(
            integration_nodes=3 ** len(variable_parameters),
            visualizations=False,
            return_raw_samples=True,
            verbose=True,
        )[0]
        acquisition_function = qNegIntegratedPosteriorVariance(
            sober_wrapper.surrogate_model, raw_taken_samples
        )
        sober_wrapper.run_SOBER(
            sober_iterations=1,
            model_samples_per_iteration=48,
            acquisition_function=lambda x: acquisition_function(x.unsqueeze(1)),
            visualizations=False,
            verbose=True,
        )
    basq_results = sober_wrapper.run_BASQ(
        integration_nodes=3 ** len(variable_parameters),
        visualizations=False,
        return_raw_samples=True,
        verbose=True,
    )
    return sober_wrapper, basq_results


def visualize_automated_assessment(
    sober_wrapper,
    basq_results,
    frequencies,
    cell_name,
    prior_name,
    soc,
    lithiation,
    electrolyte,
    variable_parameters,
):
    (
        raw_taken_samples,
        MAP,
        best_observed,
        log_expected_marginal_likelihood,
        log_approx_variance_marginal_likelihood,
    ) = basq_results
    filename = (
        cell_name
        + "_"
        + prior_name
        + "_soc_"
        + f"{soc:.3g}"
        + "_"
        + ("lithiation" if lithiation else "delithiation")
    )
    with open(filename + "_posterior.json", "w") as f:
        json.dump(
            {
                "variable_parameters": variable_parameters,
                "diagonalization": sober_wrapper.diagonalization.tolist(),
                "bounds": sober_wrapper.bounds.tolist(),
                "diag_order": sober_wrapper.diag_order,
                "raw_taken_samples": raw_taken_samples.numpy().tolist(),
            },
            f,
        )
    kde_posterior = gaussian_kde(raw_taken_samples.numpy().T)
    raw_posterior_resamples = kde_posterior.resample(32)
    posterior_resamples_pdf = kde_posterior(raw_posterior_resamples)
    alphas = posterior_resamples_pdf * (0.5 / posterior_resamples_pdf.max())
    posterior_resamples = sober_wrapper.reverse_transform(
        sober_wrapper.denormalize_input(tensor(raw_posterior_resamples.T))
    )
    parallel_arguments = [
        (trial, frequencies, cell_name, soc, electrolyte, variable_parameters)
        for trial in posterior_resamples
    ]
    with Pool() as p:
        simulations = p.starmap(simulator, parallel_arguments)

    fig_imp, ax_imp = plt.subplots(figsize=(4 * 2**0.5, 4))
    if electrolyte:
        title_text = cell_name + " DFN fit"
    else:
        title_text = cell_name + " SPM fit"
    nyquist_plot(
        fig_imp,
        ax_imp,
        frequencies.detach().numpy(),
        impedances.detach().numpy(),
        ls=":",
        lw=2,
        title_text=title_text,
        legend_text="experiment",
    )
    nyquist_plot(
        fig_imp,
        ax_imp,
        frequencies.detach().numpy(),
        simulator(MAP, frequencies, cell_name, soc, electrolyte, variable_parameters),
        title_text=title_text,
        legend_text="optimal fit",
        add_frequency_colorbar=False,
    )
    for simulated_impedances, alpha in zip(simulations, alphas):
        nyquist_plot(
            fig_imp,
            ax_imp,
            frequencies.detach().numpy(),
            simulated_impedances,
            lw=1,
            alpha=alpha,
            title_text=title_text,
            legend_text=None,
            add_frequency_colorbar=False,
        )
    if cell_name == "18650_LG_3500_MJ1_EIS_anode_discharge":
        ax_imp.set_xlim(0.8, 4.8)
        ax_imp.set_ylim(0.0, 2.0)
    """
    elif cell_name == "monolayer_17_microns":
        ax_imp.set_xlim(0, 125)
        ax_imp.set_ylim(0, 83)
    elif cell_name == "porous_42_microns":
        ax_imp.set_xlim(0, 12.9)
        ax_imp.set_ylim(0, 18.7)
    elif cell_name == "porous_80_microns":
        ax_imp.set_xlim(0, 12.9)
        ax_imp.set_ylim(0, 12.9)
    """
    fig_imp.savefig(filename + ".pdf", bbox_inches="tight", pad_inches=0.0)
    plt.show()


if __name__ == "__main__":
    ##########################################################
    # Select which cell to parameterize on which data point. #
    ##########################################################

    # One of 'monolayer_17_microns', 'porous_42_microns', 'porous_80_microns',
    # or '18650_LG_3500_MJ1_EIS_anode_discharge' (recommendation: soc_index 3).
    cell_name = "porous_80_microns"
    # Choose the parameter prior.
    prior_name = "pouch_SPM"

    soc_index = 6 if "microns" in cell_name else 3
    # Select the (de-)lithiation impedance measurement at that point.
    # (Not applicable for the delithiation-only LG-MJ1 measurement.)
    lithiation = True

    ######################################
    # Read in model parameters and data. #
    ######################################

    electrolyte = False if "SPM" in prior_name else True
    if cell_name == "18650_LG_3500_MJ1_EIS_anode_discharge":
        parameters = kuhn.get_parameters_with_switched_electrodes()
        sei = True
        working_electrode = "both"
        three_electrode = "positive"
        reference_electrode_location = 0.5
    else:
        parameters = gunther.get_parameters_for_cell(cell_name)
        sei = False
        working_electrode = "positive"
        three_electrode = "positive"
        reference_electrode_location = 1.0
    if "variant_2" in prior_name:
        parameters.update(
            {
                "Positive electrode double-layer capacity [F.m-2]": 3e-2,
                "SEI relative permittivity": 20.0,
            }
        )

    parameter_range = parameter_ranges[prior_name]
    parameter_transform = parameter_transforms[prior_name]
    parameter_transform_numpy = parameter_transforms_numpy[prior_name]

    variable_parameters = list(parameter_range.keys())
    data_index = (
        soc_index
        if cell_name == "18650_LG_3500_MJ1_EIS_anode_discharge"
        else soc_index * 2 - 1 + lithiation
    )
    electrode = (
        "negative"
        if cell_name == "18650_LG_3500_MJ1_EIS_anode_discharge"
        else "positive"
    )
    location = 0.5 if cell_name == "18650_LG_3500_MJ1_EIS_anode_discharge" else 1.0
    frequencies, impedances, soc = preprocess_data(data_index, electrode, cell_name)
    parameters[electrode.capitalize() + " electrode SOC"] = soc

    with open("impedance_ocv_alignment.json") as f:
        ocv_alignment = json.load(f)[cell_name]
    raw_data_index = ocv_alignment["indices"].index(data_index)
    with open("drt_finetuning.json") as f:
        drt_settings = json.load(f)
        lambda_value = drt_settings[cell_name]["lambda"][raw_data_index]
        subsampling = drt_settings[cell_name]["subsampling"][raw_data_index]
        start_frequency = drt_settings[cell_name]["start_frequency"][raw_data_index]
        end_frequency = drt_settings[cell_name]["end_frequency"][raw_data_index]
    """
    drt_finetuning = interactive_drt_finetuning(
        frequencies.detach().cpu().numpy(),
        impedances.detach().cpu().numpy(),
        lambda_value,
        subsampling,
        start_frequency,
        end_frequency,
    )
    lambda_value = drt_finetuning["lambda"]
    subsampling = drt_finetuning["subsampling"]
    start_frequency = drt_finetuning["start"]
    end_frequency = drt_finetuning["end"]
    """

    start = find_occurrences(frequencies, start_frequency)[0]
    end = find_occurrences(frequencies, end_frequency)[0]
    optimizer_frequencies = frequencies[start : end + 1 : subsampling]
    optimizer_impedances = impedances[start : end + 1 : subsampling]

    unknowns = {key: parameter_range[key] for key in variable_parameters}
    transform_unknowns = {
        key: (parameter_transform_numpy[key][1], parameter_transform_numpy[key][0])
        for key in variable_parameters
    }

    # Make sure that no function parameters try to get passed as numbers.
    for key in variable_parameters:
        if "function" in str(type(parameters[key])):
            forward = transform_unknowns[key][0]
            backward = transform_unknowns[key][1]
            parameters[key] = forward(
                0.5 * (backward(unknowns[key][0]) + backward(unknowns[key][1]))
            )

    mean = tensor([parameters[key] for key in variable_parameters])
    bounds = tensor(
        [
            [parameter_range[key][0] for key in variable_parameters],
            [parameter_range[key][1] for key in variable_parameters],
        ]
    )
    transforms = [parameter_transform[key] for key in variable_parameters]

    ######################################
    # Perform a manual model assessment. #
    ######################################

    """
    manual_model_assessment(
        optimizer_frequencies,
        optimizer_impedances,
        parameters,
        unknowns,
        transform_unknowns,
        working_electrode,
        electrolyte,
        sei,
        three_electrode,
        reference_electrode_location,
        lambda_value,
    )
    """

    ############################################
    # Perform the model assessment with SOBER. #
    ############################################

    filename = (
        cell_name
        + "_"
        + prior_name
        + "_soc_"
        + f"{soc:.3g}"
        + "_"
        + ("lithiation" if lithiation else "delithiation")
        + ".log"
    )
    with open(filename, "w") as f:
        with redirect_stdout(f):
            sober_wrapper, basq_results = automated_model_assessment(
                optimizer_frequencies,
                optimizer_impedances,
                mean,
                bounds,
                transforms,
                cell_name,
                soc,
                electrolyte,
                variable_parameters,
                lambda_value,
            )
            # from pandas import DataFrame
            # from seaborn import pairplot
            # pairplot(DataFrame(sober_wrapper.tm.numpy(sober_wrapper.X_all), columns=sober_wrapper.names))
            # plt.show()
            visualize_automated_assessment(
                sober_wrapper,
                basq_results,
                frequencies,
                cell_name,
                prior_name,
                soc,
                lithiation,
                electrolyte,
                variable_parameters,
            )
