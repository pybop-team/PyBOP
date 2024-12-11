import numpy as np
import pybop
import pybamm
from scipy.io import savemat
import time as timer
import matplotlib.pyplot as plt

## Fixed parameters
Nruns = 1

SOC = 0.5
Nfreq = 60
fmin = 2e-4
fmax = 1e3
frequencies = np.logspace(np.log10(fmin), np.log10(fmax), Nfreq)

parameter_set = pybop.ParameterSet.pybamm("Chen2020")
parameter_set["Contact resistance [Ohm]"] = 0.01
model_options = {"surface form": "differential", "contact resistance": "true"}


## Time domain simulation
# Time domain
I_app = 100e-3
number_of_periods = 10
samples_per_period = 56


def current_function(t):
    return I_app * np.sin(2 * np.pi * pybamm.Parameter("Frequency [Hz]") * t)


parameter_set.update(
    {
        "Current function [A]": current_function,
        "Frequency [Hz]": 10,
    },
    check_already_exists=False,
)

model = pybop.lithium_ion.DFN(
    parameter_set=parameter_set,
    options=model_options,
)
var_pts = model._unprocessed_model.default_var_pts
var_pts["x_n"] = 100
var_pts["x_p"] = 100
var_pts["r_n"] = 100
var_pts["r_p"] = 100

model_time = pybop.lithium_ion.DFN(
    parameter_set=parameter_set,
    solver=pybamm.ScipySolver(atol=1e-9),
    options=model_options,
    var_pts=var_pts,
)
model_freq = pybop.lithium_ion.DFN(
    parameter_set=parameter_set,
    options=model_options,
    var_pts=var_pts,
    eis=True,
)
parameters = pybop.Parameters(pybop.Parameter("Frequency [Hz]", initial_value=10))
model_time.build(parameters=parameters, initial_state={"Initial SoC": SOC})
model_freq.build(initial_state={"Initial SoC": SOC})

## Time domain simulation
# time_elapsed = np.zeros([Nruns, 1])
# for ii in range(Nruns):
#     start_time = timer.time()

#     impedance_time = []
#     for frequency in frequencies:
#         # Solve
#         period = 1 / frequency
#         dt = period / samples_per_period
#         t_eval = np.array(range(0, samples_per_period * number_of_periods)) * dt
#         sol = model_time.simulate(
#             inputs={"Frequency [Hz]": frequency},
#             t_eval=t_eval,
#         )
#         # Extract final P periods of the solution
#         P = 5
#         time = sol["Time [s]"].entries[-P * samples_per_period :]
#         current = sol["Current [A]"].entries[-P * samples_per_period :]
#         voltage = sol["Voltage [V]"].entries[-P * samples_per_period :]

#         # FFT
#         current_fft = np.fft.fft(current)
#         voltage_fft = np.fft.fft(voltage)

#         # Get index of first harmonic
#         impedance = -voltage_fft[P] / current_fft[P]
#         impedance_time.append(impedance)

#     end_time = timer.time()
#     time_elapsed[ii] = end_time - start_time
# print("Time domain method: ", np.mean(time_elapsed), "s")

# plt.plot(np.real(impedance_time), -np.imag(impedance_time))
# plt.show()

## Frequency domain simulation
time_elapsed = np.zeros([Nruns, 1])
for ii in range(Nruns):
    start_time = timer.time()
    simulationFD = model_freq.simulateEIS(
        inputs=None,
        f_eval=frequencies,
    )
    impedance_freq = simulationFD["Impedance"]

    end_time = timer.time()
    time_elapsed[ii] = end_time - start_time
print("Frequency domain method: ", np.mean(time_elapsed), "s")

# plt.plot(np.real(impedance_freq), -np.imag(impedance_freq))
# plt.show()


# Write to matfile
# mdic = {
#     "Ztime": impedance_time,
#     "Zfreq": impedance_freq,
#     "f": frequencies,
# }
# savemat("Data comparison time freq/Z_DFN_timeFreq.mat", mdic)
