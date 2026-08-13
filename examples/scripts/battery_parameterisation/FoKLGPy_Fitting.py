import numpy as np
import pybamm

import pybop
import matplotlib.pyplot as plt

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

            "Discharge at 1C for 10 minutes",
            "Rest for 5 minutes",
            "Discharge at 2C for 3 minutes or until 3.0 V",
            "Rest for 10 minutes",
            "Charge at 1C for 5 minutes or until 4.2 V",
            "Rest for 10 minutes",

    ] * 3
)

# 2. Generate a synthetic dataset using the experiment
sim = pybamm.Simulation(model, experiment=experiment, parameter_values=parameter_values)

solution = sim.solve()

t_eval = solution["Time [s]"].entries
original = solution["Positive electrode exchange current density [A.m-2]"].entries
sigma = 5e-3
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current [A]": solution["Current [A]"].entries,
        "Voltage [V]": pybop.add_noise(solution["Voltage [V]"].entries, sigma),
        "Bulk open-circuit voltage [V]": solution["Bulk open-circuit voltage [V]"].entries,
    }
)

# Create GP terms
counter = 0
tolerance = 1
num_of_terms = 7
bic_i = 1e10

GP_options = {'Number of terms':num_of_terms,'arg_inds':[0],'Normalization min-max':{'0':(400,2500)},
              'Constant mean':3.4,'div_arg':[[1,2]], 'exp':True}

GP_param_neg = pybop.FoKLGP("Positive electrode exchange-current density [A.m-2]", parameter_values=parameter_values.copy(), options=GP_options, twoway=True)
new_parameters = GP_param_neg.get_parameter_values()


simulator = pybop.pybamm.Simulator(model, new_parameters, protocol=dataset)
target = ["Voltage [V]", "Bulk open-circuit voltage [V]"]
cost = pybop.GaussianLogLikelihoodKnownSigma(dataset,sigma=sigma, target=target)
problem = pybop.Problem(simulator, cost)

# Set up the optimiser
options = pybop.PintsOptions(
    max_iterations=100,
    max_unchanged_iterations=30,
    verbose=True
)
optim = pybop.IRPropPlus(problem, options=options)


result = optim.run()
L = result.best_cost
k = len(result.x)
n = len(dataset.data['Time [s]'])
bic = -2*L + k*np.log(n)
print(bic)


# Plot the timeseries output
pybop.plot.problem(problem, inputs=result.best_inputs, title="Optimised Comparison")


# 1. Re-initialize the simulation using the EXACT SAME experiment used for training
sim_final = pybamm.Simulation(
    model,
    experiment=experiment,  # CRITICAL: Must match the training protocol
    parameter_values=new_parameters
)

# 2. Run the final solve (let the experiment handle the time steps natively)
solution_final = sim_final.solve(inputs=result.best_inputs)

solution_final.all_inputs = [result.best_inputs] * len(solution_final.all_ys)

new = solution_final["Positive electrode exchange current density [A.m-2]"].entries
t_max = min(solution["Time [s]"].entries[-1], solution_final["Time [s]"].entries[-1])

t_eval_valid = t_eval[t_eval <= t_max]

j0_original_var = solution["Positive electrode exchange current density [A.m-2]"]
j0_new_var = solution_final["Positive electrode exchange current density [A.m-2]"]

original_aligned = j0_original_var(t=t_eval_valid)
new_aligned = j0_new_var(t=t_eval_valid)


idx_top = 0
idx_mid = new_aligned.shape[0] // 2
idx_bot = -1

gp_top = new_aligned[idx_top, :]
gp_mid = new_aligned[idx_mid, :]
gp_bot = new_aligned[idx_bot, :]

ref_top = original_aligned[idx_top, :]
ref_mid = original_aligned[idx_mid, :]
ref_bot = original_aligned[idx_bot, :]

import plotly.graph_objects as go

fig = go.Figure()

# 1. Plot the Analytical Baseline (Chen2020) as dashed lines
fig.add_trace(go.Scatter(x=t_eval_valid, y=ref_top, name='Chen2020 - Near Separator',
                         line=dict(color='blue', dash='dash'), opacity=0.7))
fig.add_trace(go.Scatter(x=t_eval_valid, y=ref_mid, name='Chen2020 - Middle',
                         line=dict(color='orange', dash='dash'), opacity=0.7))
fig.add_trace(go.Scatter(x=t_eval_valid, y=ref_bot, name='Chen2020 - Near Collector',
                         line=dict(color='green', dash='dash'), opacity=0.7))

# 2. Plot the GP Estimate as solid lines
fig.add_trace(go.Scatter(x=t_eval_valid, y=gp_top, name='GP - Near Separator',
                         line=dict(color='blue')))
fig.add_trace(go.Scatter(x=t_eval_valid, y=gp_mid, name='GP - Middle',
                         line=dict(color='orange')))
fig.add_trace(go.Scatter(x=t_eval_valid, y=gp_bot, name='GP - Near Collector',
                         line=dict(color='green')))

# 3. Formatting for readability
fig.update_layout(
    title='Positive Electrode J0 Fitting',
    xaxis_title='Time [s]',
    yaxis_title='J0 [A.m-2]',
    template='plotly_white',
    # Place legend outside the plot
    legend=dict(x=1.02, y=0.5)
)

# 4. Use a dotted grid
fig.update_xaxes(showgrid=True, griddash='dot')
fig.update_yaxes(showgrid=True, griddash='dot')

fig.show()