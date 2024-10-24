import matplotlib.pyplot as plt
import numpy as np
import pybop
from pybamm.input.parameters.lithium_ion.Chen2020 import (
    graphite_LGM50_ocp_Chen2020,
    nmc_LGM50_ocp_Chen2020,
)
from pybamm.models.full_battery_models.lithium_ion.electrode_soh import get_min_max_stoichiometries
from scipy.io import savemat


# Unpack parameter values from Chen2020
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
parameter_set["Contact resistance [Ohm]"] = 0.01

F = parameter_set["Faraday constant [C.mol-1]"]
T = parameter_set["Ambient temperature [K]"]
alpha_p = parameter_set["Positive electrode active material volume fraction"]
alpha_n = parameter_set["Negative electrode active material volume fraction"]
c_max_p = parameter_set["Maximum concentration in positive electrode [mol.m-3]"]
c_max_n = parameter_set["Maximum concentration in negative electrode [mol.m-3]"]
L_p = parameter_set["Positive electrode thickness [m]"]
L_n = parameter_set["Negative electrode thickness [m]"]
epsilon_p = parameter_set["Positive electrode porosity"]
epsilon_n = parameter_set["Negative electrode porosity"]
R_p = parameter_set["Positive particle radius [m]"]
R_n = parameter_set["Negative particle radius [m]"]
D_p = parameter_set["Positive particle diffusivity [m2.s-1]"]
D_n = parameter_set["Negative particle diffusivity [m2.s-1]"]
b_p = parameter_set["Positive electrode Bruggeman coefficient (electrolyte)"]
b_n = parameter_set["Negative electrode Bruggeman coefficient (electrolyte)"]
Cdl_p = parameter_set["Positive electrode double-layer capacity [F.m-2]"]
Cdl_n = parameter_set["Negative electrode double-layer capacity [F.m-2]"]
m_p = 3.42e-6  # (A/m2)(m3/mol)**1.5
m_n = 6.48e-7  # (A/m2)(m3/mol)**1.5

A = parameter_set["Electrode height [m]"] * parameter_set["Electrode width [m]"]
L = L_p + L_n + parameter_set["Separator thickness [m]"]
ce0 = parameter_set["Initial concentration in electrolyte [mol.m-3]"]
De = parameter_set["Electrolyte diffusivity [m2.s-1]"](ce0, T)
epsilon_sep = parameter_set["Separator porosity"]
b_sep = parameter_set["Separator Bruggeman coefficient (electrolyte)"]
t_plus = parameter_set["Cation transference number"]
R0 = parameter_set["Contact resistance [Ohm]"]

# Compute the stoichiometry limits and initial SOC
x_0, x_100, y_100, y_0 = get_min_max_stoichiometries(parameter_set)

# Grouped parameters
Q_th_p = F * alpha_p * c_max_p * L_p * A
Q_th_n = F * alpha_n * c_max_n * L_n * A

beta_p = alpha_p * c_max_p / (epsilon_p * ce0)
beta_n = alpha_n * c_max_n / (epsilon_n * ce0)

tau_d_p = R_p**2 / D_p
tau_d_n = R_n**2 / D_n

tau_e_p = L**2 / (epsilon_p ** (b_p - 1) * De)
tau_e_n = L**2 / (epsilon_n ** (b_n - 1) * De)
tau_e_sep = L**2 / (epsilon_sep ** (b_sep - 1) * De)

tau_ct_p = F * R_p * L_p / (m_p * L * np.sqrt(ce0))
tau_ct_n = F * R_n * L_n / (m_n * L * np.sqrt(ce0))

C_p = Cdl_p * alpha_p * L_p * A / R_p
C_n = Cdl_n * alpha_n * L_n * A / R_n

l_p = L_p / L
l_n = L_n / L

grouped_parameter_set = {
    "Current function [A]": parameter_set["Current function [A]"],
    "Nominal cell capacity [A.h]": parameter_set["Nominal cell capacity [A.h]"],
    "Initial temperature [K]": T,
    "Initial SoC": 0.5,
    "Minimum negative stoichiometry": x_0,
    "Maximum negative stoichiometry": x_100,
    "Minimum positive stoichiometry": y_100,
    "Maximum positive stoichiometry": y_0,
    "Lower voltage cut-off [V]": parameter_set["Lower voltage cut-off [V]"],
    "Upper voltage cut-off [V]": parameter_set["Upper voltage cut-off [V]"],
    "Positive electrode thickness [m]": l_p,  # normalised
    "Negative electrode thickness [m]": l_n,  # normalised
    "Separator thickness [m]": 1-l_p-l_n,  # normalised
    "Positive particle radius [m]": 1,  # normalised
    "Negative particle radius [m]": 1,  # normalised
    "Positive electrode OCP [V]": nmc_LGM50_ocp_Chen2020,
    "Negative electrode OCP [V]": graphite_LGM50_ocp_Chen2020,
    "Positive theoretical electrode capacity [As]": Q_th_p,
    "Negative theoretical electrode capacity [As]": Q_th_n,
    "Positive relative concentration": beta_p,
    "Negative relative concentration": beta_n,
    "Positive particle diffusion time scale [s]": tau_d_p,
    "Negative particle diffusion time scale [s]": tau_d_n,
    "Positive electrode electrolyte diffusion time scale [s]": tau_e_p,
    "Negative electrode electrolyte diffusion time scale [s]": tau_e_n,
    "Separator electrolyte diffusion time scale [s]": tau_e_sep,
    "Positive electrode charge transfer time scale [s]": tau_ct_p,
    "Negative electrode charge transfer time scale [s]": tau_ct_n,
    "Positive electrode capacitance [F]": C_p,
    "Negative electrode capacitance [F]": C_n,
    "Cation transference number": t_plus,
    "Positive electrode relative thickness": l_p,
    "Negative electrode relative thickness": l_n,
    "Series resistance [Ohm]": R0,
}

# Test model in the time domain
model_options = {"contact resistance": "true"}
time_domain_SPMe = pybop.lithium_ion.SPMe(parameter_set=parameter_set, options=model_options)
time_domain_grouped = pybop.lithium_ion.GroupedSPMe(parameter_set=grouped_parameter_set)
for model in [time_domain_SPMe, time_domain_grouped]:
    model.build(initial_state={"Initial SoC": 0.9})
    simulation = model.predict(t_eval=np.linspace(0,3600,100))
    dataset = pybop.Dataset(
        {
            "Time [s]": simulation["Time [s]"].data,
            "Current function [A]": simulation["Current [A]"].data,
            "Voltage [V]": simulation["Voltage [V]"].data,
        }
    )
    pybop.plot.dataset(dataset)

# Continue with frequency domain model
freq_domain_SPMe = pybop.lithium_ion.SPMe(parameter_set=parameter_set, options=model_options, eis=True)
freq_domain_grouped = pybop.lithium_ion.GroupedSPMe(parameter_set=grouped_parameter_set, eis=True)

for model in [freq_domain_SPMe, freq_domain_grouped]:
    NSOC = 11
    Nfreq = 60
    fmin = 4e-4
    fmax = 1e3
    SOCs = np.linspace(0, 1, NSOC)
    frequencies = np.logspace(np.log10(fmin), np.log10(fmax), Nfreq)

    impedances = 1j * np.zeros((Nfreq, NSOC))
    for ii, SOC in enumerate(SOCs):
        model.build(initial_state={"Initial SoC": SOC})
        simulation = model.simulateEIS(inputs=None, f_eval=frequencies)
        impedances[:, ii] = simulation["Impedance"]

    fig, ax = plt.subplots()
    for ii in range(len(SOCs)):
        ax.plot(np.real(impedances[:, ii]), -np.imag(impedances[:, ii]))
    ax.set(xlabel="$Z_r(\omega)$ [$\Omega$]", ylabel="$-Z_j(\omega)$ [$\Omega$]")
    ax.grid()
    ax.set_aspect("equal", "box")
    plt.show()

fig.savefig("Nyquist.png")

# mdic = {"Z": impedances, "f": frequencies, "SOC": SOCs}
# savemat("Simulated data SPMe/Z_SPMe_SOC_Pybop_chen2020.mat", mdic)
