import plotly.graph_objects as go
import pybamm

import pybop

"""
In this example, we demonstrate the use of FoKL decomposed Gaussian Processes for estimation of
some parameters.
"""


# Define model and parameter values
model = pybamm.lithium_ion.DFN()

parameter_values = pybamm.ParameterValues("Chen2020")

# 1. Define a dynamic pulse experiment to excite spatial concentration gradients

experiment = pybamm.Experiment(
    [
        "Discharge at 2C until 2.5 V",
    ]
)


experiment_validate = pybamm.Experiment(
    [
        "Discharge at 3C until 2.7 V",
    ]
)

# 2. Generate a synthetic dataset using the experiment
sim = pybamm.Simulation(model, experiment=experiment, parameter_values=parameter_values)
sim_validate = pybamm.Simulation(
    model, experiment=experiment_validate, parameter_values=parameter_values
)

solution = sim.solve()
solution_validate = sim_validate.solve()

t_eval_validate = solution_validate["Time [s]"].entries
t_eval = solution["Time [s]"].entries

sigma = 5e-3

# Construct a PyBOP dataset for training and validation
V_train_data = pybop.add_noise(solution["Voltage [V]"].entries, sigma)
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current [A]": solution["Current [A]"].entries,
        "Voltage [V]": V_train_data,
        "Bulk open-circuit voltage [V]": solution[
            "Bulk open-circuit voltage [V]"
        ].entries,
    }
)

V_data = pybop.add_noise(solution_validate["Voltage [V]"].entries, sigma)
dataset_validate = pybop.Dataset(
    {
        "Time [s]": t_eval_validate,
        "Current [A]": solution_validate["Current [A]"].entries,
        "Voltage [V]": V_data,
        "Bulk open-circuit voltage [V]": solution_validate[
            "Bulk open-circuit voltage [V]"
        ].entries,
    }
)


# Create GP terms

num_of_terms = 3

GP_options = {
    "Number of terms": num_of_terms,
    "Arguments": [
        "Electrolyte concentration [mol.m-3]"
    ],  # Argument [0] corresponds to concentration in the electrolyte
    "Normalization min-max": {
        "Electrolyte concentration [mol.m-3]": (-1, 5000)
    },  # Normalizing such that inputs are between 0-1 is necessary
    "Constant mean": 1.8e-10,  # Beta 0 mean
    "Constant standard deviation": 1e-11,
    "Bi standard deviation": 5e-11,
    "exp": False,
    "Verbose": True,
}

GP_param_neg = pybop.FoKLGP(
    "Electrolyte diffusivity [m2.s-1]",
    parameter_values=parameter_values.copy(),
    options=GP_options,
    twoway=True,
    model=model,
)
new_parameters = GP_param_neg.get_parameter_values()

# define constant parameter estimation
new_parameters_constant = parameter_values.copy()
new_parameters_constant.update(
    {
        "Electrolyte diffusivity [m2.s-1]": pybop.Parameter(
            pybop.Gaussian(1.79e-10, 2e-11, truncated_at=[1e-14, 1e-7])
        )
    }
)

simulator = pybop.pybamm.Simulator(model, new_parameters, protocol=dataset)
target = ["Voltage [V]", "Bulk open-circuit voltage [V]"]
cost = pybop.GaussianLogLikelihoodKnownSigma(dataset, sigma=sigma, target=target)
problem = pybop.Problem(simulator, cost)

simulator_constant = pybop.pybamm.Simulator(
    model, new_parameters_constant, protocol=dataset
)
problem_constant = pybop.Problem(simulator_constant, cost)


# Set up the optimiser
options = pybop.PintsOptions(
    max_iterations=1000,
    max_unchanged_iterations=100,
    verbose=True,
)
optim_GP = pybop.XNES(problem, options=options)

result = optim_GP.run()
optim_constant = pybop.XNES(problem_constant, options=options)


result_constant = optim_constant.run()


new_parameters_constant.update(result_constant.best_inputs)

sim_final_train_constant = pybamm.Simulation(
    model, experiment=experiment, parameter_values=new_parameters_constant
)

sim_final_train = pybamm.Simulation(
    model, experiment=experiment, parameter_values=new_parameters
)

sim_final_test_constant = pybamm.Simulation(
    model, experiment=experiment_validate, parameter_values=new_parameters_constant
)

sim_final_test = pybamm.Simulation(
    model, experiment=experiment_validate, parameter_values=new_parameters
)

# Run the final solve
solution_final_train = sim_final_train.solve(inputs=result.best_inputs)
solution_final_train_constant = sim_final_train_constant.solve()

solution_final_test = sim_final_test.solve(inputs=result.best_inputs)
solution_final_test_constant = sim_final_test_constant.solve()

V_test = solution_final_test["Voltage [V]"].entries
V_train = solution_final_train["Voltage [V]"].entries
t_eval_validate_GP = solution_final_test["Time [s]"].entries
t_eval_validate_GP_train = solution_final_train["Time [s]"].entries

V_test_constant = solution_final_test_constant["Voltage [V]"].entries
V_train_constant = solution_final_train_constant["Voltage [V]"].entries
t_eval_validate_constant = solution_final_test_constant["Time [s]"].entries
t_eval_validate_constant_train = solution_final_train_constant["Time [s]"].entries

# --- Validation experiment (3C discharge) ---
fig_val = go.Figure()
fig_val.add_trace(
    go.Scatter(
        x=t_eval_validate, y=V_data, mode="markers", name="Data", marker=dict(size=4)
    )
)
fig_val.add_trace(
    go.Scatter(
        x=t_eval_validate_GP,
        y=V_test,
        mode="lines",
        name="FoKL GP",
        line=dict(color="green"),
    )
)
fig_val.add_trace(
    go.Scatter(
        x=t_eval_validate_constant,
        y=V_test_constant,
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

# --- Training experiment (2C discharge) ---
fig_train = go.Figure()
fig_train.add_trace(
    go.Scatter(
        x=t_eval, y=V_train_data, mode="markers", name="Data", marker=dict(size=4)
    )
)
fig_train.add_trace(
    go.Scatter(
        x=t_eval_validate_GP_train,
        y=V_train,
        mode="lines",
        name="FoKL GP",
        line=dict(color="green"),
    )
)
fig_train.add_trace(
    go.Scatter(
        x=t_eval_validate_constant_train,
        y=V_train_constant,
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

# Plot the optimisation result
result.plot_convergence()
result.plot_parameters()
