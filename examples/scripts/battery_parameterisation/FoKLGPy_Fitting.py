import numpy as np
import plotly.graph_objects as go
import pybamm

import pybop

"""
In this example, we demonstrate the use of FoKL decomposed Gaussian Processes
for state-dependent estimation of electrolyte diffusivity.
"""

# 1. Define model, base parameters, and experiments
model = pybamm.lithium_ion.DFN()
parameter_values = pybamm.ParameterValues("Chen2020")

experiment_train = pybamm.Experiment(["Discharge at 2C until 2.5 V"])
experiment_validate = pybamm.Experiment(["Discharge at 3C until 2.7 V"])

# 2. Generate synthetic training and validation datasets
sim_train = pybamm.Simulation(
    model, experiment=experiment_train, parameter_values=parameter_values
)
sim_validate = pybamm.Simulation(
    model, experiment=experiment_validate, parameter_values=parameter_values
)

solution_train = sim_train.solve()
solution_validate = sim_validate.solve()

t_train = solution_train["Time [s]"].entries
t_validate = solution_validate["Time [s]"].entries

sigma = 5e-3
V_train_noisy = pybop.add_noise(solution_train["Voltage [V]"].entries, sigma)
V_validate_noisy = pybop.add_noise(solution_validate["Voltage [V]"].entries, sigma)

dataset_train = pybop.Dataset(
    {
        "Time [s]": t_train,
        "Current [A]": solution_train["Current [A]"].entries,
        "Voltage [V]": V_train_noisy,
    }
)

dataset_validate = pybop.Dataset(
    {
        "Time [s]": t_validate,
        "Current [A]": solution_validate["Current [A]"].entries,
        "Voltage [V]": V_validate_noisy,
    }
)

# 3. Configure FoKL GP parameter representation
# Using exp=True guarantees D_e(c_e) > 0 during optimization
GP_options = {
    "Number of terms": 3,
    "Arguments": ["Electrolyte concentration [mol.m-3]"],
    "Normalization min-max": {"Electrolyte concentration [mol.m-3]": (0.0, 3000.0)},
    "Constant mean": np.log(1.8e-10),
    "Constant standard deviation": 0.2,
    "Bi mean": 0.0,
    "Bi standard deviation": 0.2,
    "exp": True,
    "Verbose": True,
}

GP_param = pybop.FoKLGP(
    "Electrolyte diffusivity [m2.s-1]",
    parameter_values=parameter_values.copy(),
    options=GP_options,
    twoway=True,
    model=model,
)
parameters_gp = GP_param.get_parameter_values()

# 4. Configure constant scalar parameter representation for comparison
parameters_constant = parameter_values.copy()
parameters_constant.update(
    {
        "Electrolyte diffusivity [m2.s-1]": pybop.Parameter(
            pybop.Gaussian(1.79e-10, 2e-11, truncated_at=[1e-14, 1e-7])
        )
    }
)

# 5. Build PyBOP Simulators, Cost, and Problems
target = ["Voltage [V]"]
cost = pybop.GaussianLogLikelihoodKnownSigma(dataset_train, sigma=sigma, target=target)

simulator_gp = pybop.pybamm.Simulator(model, parameters_gp, protocol=dataset_train)
problem_gp = pybop.Problem(simulator_gp, cost)

simulator_constant = pybop.pybamm.Simulator(
    model, parameters_constant, protocol=dataset_train
)
problem_constant = pybop.Problem(simulator_constant, cost)

# 6. Run XNES Optimization
options = pybop.PintsOptions(
    max_iterations=250,
    max_unchanged_iterations=50,
    verbose=True,
)

optim_gp = pybop.XNES(problem_gp, options=options)
result_gp = optim_gp.run()

optim_constant = pybop.XNES(problem_constant, options=options)
result_constant = optim_constant.run()

# 7. Evaluate optimized parameters on Training (2C) and Validation (3C) experiments
# Use fresh ParameterValues copies so PyBaMM simulations run cleanly
eval_params_gp = GP_param.get_parameter_values().copy()
eval_params_constant = parameter_values.copy()
eval_params_constant.update(result_constant.best_inputs)

sim_final_train_gp = pybamm.Simulation(
    model, experiment=experiment_train, parameter_values=eval_params_gp
)
sim_final_test_gp = pybamm.Simulation(
    model, experiment=experiment_validate, parameter_values=eval_params_gp
)

sim_final_train_const = pybamm.Simulation(
    model, experiment=experiment_train, parameter_values=eval_params_constant
)
sim_final_test_const = pybamm.Simulation(
    model, experiment=experiment_validate, parameter_values=eval_params_constant
)

sol_train_gp = sim_final_train_gp.solve(inputs=result_gp.best_inputs)
sol_test_gp = sim_final_test_gp.solve(inputs=result_gp.best_inputs)

sol_train_const = sim_final_train_const.solve()
sol_test_const = sim_final_test_const.solve()

# 8. Plot Validation (3C discharge) comparison
fig_val = go.Figure()
fig_val.add_trace(
    go.Scatter(
        x=t_validate,
        y=V_validate_noisy,
        mode="markers",
        name="Data",
        marker=dict(size=4),
    )
)
fig_val.add_trace(
    go.Scatter(
        x=sol_test_gp["Time [s]"].entries,
        y=sol_test_gp["Voltage [V]"].entries,
        mode="lines",
        name="FoKL GP",
        line=dict(color="green"),
    )
)
fig_val.add_trace(
    go.Scatter(
        x=sol_test_const["Time [s]"].entries,
        y=sol_test_const["Voltage [V]"].entries,
        mode="lines",
        name="PyBOP Constant",
        line=dict(color="red"),
    )
)
fig_val.update_layout(
    xaxis_title="Time [s]",
    yaxis_title="Voltage [V]",
    title="Validation experiment (3C discharge)",
    legend=dict(x=0.01, y=0.01),
)
fig_val.show()

# 9. Plot Training (2C discharge) comparison
fig_train = go.Figure()
fig_train.add_trace(
    go.Scatter(
        x=t_train,
        y=V_train_noisy,
        mode="markers",
        name="Data",
        marker=dict(size=4),
    )
)
fig_train.add_trace(
    go.Scatter(
        x=sol_train_gp["Time [s]"].entries,
        y=sol_train_gp["Voltage [V]"].entries,
        mode="lines",
        name="FoKL GP",
        line=dict(color="green"),
    )
)
fig_train.add_trace(
    go.Scatter(
        x=sol_train_const["Time [s]"].entries,
        y=sol_train_const["Voltage [V]"].entries,
        mode="lines",
        name="PyBOP Constant",
        line=dict(color="red"),
    )
)
fig_train.update_layout(
    xaxis_title="Time [s]",
    yaxis_title="Voltage [V]",
    title="Training experiment (2C discharge)",
    legend=dict(x=0.01, y=0.01),
)
fig_train.show()

# Plot optimisation diagnostics
result_gp.plot_convergence()
result_gp.plot_parameters()
