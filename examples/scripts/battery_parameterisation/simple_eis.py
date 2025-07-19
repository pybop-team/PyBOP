import numpy as np
import pybamm

import pybop

# Define model
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values["Contact resistance [Ohm]"] = 0.0
initial_state = {"Initial SoC": 0.5}
n_frequency = 20
sigma0 = 1e-4
f_eval = np.logspace(-4, 5, n_frequency)
model = pybamm.lithium_ion.SPM(
    options={"surface form": "differential", "contact resistance": "true"},
)

# Fitting parameters
parameters = pybop.Parameters(
    [
        pybop.Parameter(
            "Positive particle diffusivity [m2.s-1]",
            prior=pybop.Uniform(2e-15, 6e-15),
            bounds=[2e-15, 6e-15],
        ),
        pybop.Parameter(
            "Contact resistance [Ohm]",
            prior=pybop.Uniform(0, 0.05),
            bounds=[0, 0.05],
        ),
    ]
)

# Create synthetic data for parameter inference
eis_pipeline = pybop.pipelines.PybammEISPipeline(
    model,
    f_eval=f_eval,
    parameter_values=parameter_values.copy(),
    pybop_parameters=parameters,
    initial_state=initial_state,
)
eis_pipeline.build()
eis_pipeline.pybop_parameters.update(values=np.asarray([3.31e-15, 0.0232]))
eis_pipeline.rebuild()
impedance = eis_pipeline.solve()


def noisy(data, sigma):
    # Generate real part noise
    real_noise = np.random.normal(0, sigma, len(data))

    # Generate imaginary part noise
    imag_noise = np.random.normal(0, sigma, len(data))

    # Combine them into a complex noise
    return data + real_noise + 1j * imag_noise


# Form dataset
dataset = pybop.Dataset(
    {
        "Frequency [Hz]": f_eval,
        "Current function [A]": np.ones(n_frequency) * 0.0,
        "Impedance": noisy(impedance, sigma0),
    }
)

# Pass the initial state to avoid rebuilding
for c in [
    "Initial concentration in negative electrode [mol.m-3]",
    "Initial concentration in positive electrode [mol.m-3]",
]:
    parameter_values[c] = eis_pipeline.pybamm_pipeline.parameter_values[c]

# Create an EIS problem
builder = pybop.PybammEIS()
builder.set_simulation(model, parameter_values=parameter_values)
builder.set_dataset(dataset)
for p in parameters:
    builder.add_parameter(p)
builder.add_cost(pybop.MeanAbsoluteError())
problem = builder.build()

# Select optimiser and run
optim = pybop.ScipyDifferentialEvolution(problem)
results = optim.run()
print(results)

# Plot the nyquist
pybop.plot.nyquist(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot 2d landscape
pybop.plot.surface(optim)
