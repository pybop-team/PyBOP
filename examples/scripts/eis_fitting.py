import numpy as np

import pybop

# Define model
parameter_set = pybop.ParameterSet.pybamm("Chen2020")
parameter_set["Contact resistance [Ohm]"] = 0.0
model = pybop.lithium_ion.SPM(
    parameter_set=parameter_set,
    options={"surface form": "differential", "contact resistance": "true"},
)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Positive electrode thickness [m]",
        prior=pybop.Gaussian(60e-6, 1e-6),
        bounds=[10e-6, 80e-6],
    ),
    pybop.Parameter(
        "Negative electrode thickness [m]",
        prior=pybop.Gaussian(40e-6, 1e-6),
        bounds=[10e-6, 80e-6],
    ),
)

# Form dataset
dataset = pybop.Dataset(
    {
        "Frequency [Hz]": np.logspace(-4, 5, 30),
        "Current function [A]": np.ones(30) * 0.0,
        "Impedance": np.asarray(  # [0.74, 0.42]
            [
                1.22932096e-01 - 1.61334852e-01j,
                1.20183857e-01 - 8.62168750e-02j,
                1.13263444e-01 - 5.18047726e-02j,
                1.04052155e-01 - 3.35022824e-02j,
                9.64515013e-02 - 2.23021233e-02j,
                9.06786107e-02 - 1.52884662e-02j,
                8.63208234e-02 - 1.06615416e-02j,
                8.31090848e-02 - 7.35620839e-03j,
                8.10200368e-02 - 4.85799774e-03j,
                8.00077590e-02 - 3.25064001e-03j,
                7.96461450e-02 - 2.82671946e-03j,
                7.94561760e-02 - 3.80233438e-03j,
                7.90211481e-02 - 6.75677894e-03j,
                7.73396194e-02 - 1.30368488e-02j,
                7.10702300e-02 - 2.41923459e-02j,
                5.33354614e-02 - 3.65349286e-02j,
                2.70601682e-02 - 3.59150273e-02j,
                1.04187830e-02 - 2.35001901e-02j,
                4.46828406e-03 - 1.31671362e-02j,
                2.19296862e-03 - 7.49687043e-03j,
                8.55961473e-04 - 4.24936512e-03j,
                2.47416663e-04 - 2.22977873e-03j,
                6.24540680e-05 - 1.11430731e-03j,
                1.51553128e-05 - 5.48240587e-04j,
                3.64126863e-06 - 2.68650763e-04j,
                8.72757867e-07 - 1.31515901e-04j,
                2.09065980e-07 - 6.43673754e-05j,
                5.00740475e-08 - 3.15013181e-05j,
                1.19929932e-08 - 1.54164989e-05j,
                2.87236101e-09 - 7.54468952e-06j,
            ]
        ),
    }
)

signal = ["Impedance"]
# Generate problem, cost function, and optimisation class
problem = pybop.EISProblem(model, parameters, dataset, signal=signal)
cost = pybop.SumSquaredError(problem)
optim = pybop.CMAES(cost, max_iterations=100, max_unchanged_iterations=30)

x, final_cost = optim.run()
print("Estimated parameters:", x)

# Plot the nyquist
pybop.nyquist(problem, problem_inputs=x, title="Optimised Comparison")

# Plot convergence
pybop.plot_convergence(optim)

# Plot the parameter traces
pybop.plot_parameters(optim)

# Plot 2d landscape
pybop.plot2d(optim, steps=10)
