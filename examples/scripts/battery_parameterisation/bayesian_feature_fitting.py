import numpy as np
import pybamm
from ep_bolfi.models.solversetup import spectral_mesh_pts_and_method

import pybop

"""
This example demonstrates how to use EP-BOLFI to parameterise a PyBaMM model using
a "feature"-based cost function. We use the term "feature" to describe a parameter
obtained from fitting either the data or a candidate solution to a simpler model.
Every evaluation of a feature-based cost function runs its own optimisation (based
on the simpler model) to identify the value of the feature. The aim is to minimise
the "feature distance" to identify the parameter values which produce a candidate
solution with a feature value as close as possible to that of the data.
"""

# Define model and parameter values
model = pybamm.lithium_ion.SPMe()
parameter_values = pybamm.ParameterValues("Chen2020")
original_D_n = parameter_values["Negative particle diffusivity [m2.s-1]"]
original_D_p = parameter_values["Positive particle diffusivity [m2.s-1]"]

# Set multivariate parameters (defined in model space)
distribution = pybop.MultivariateLogNormal(
    mean_log_x=[np.log(0.9 * original_D_n), np.log(1.1 * original_D_p)],
    covariance_log_x=[[np.log(2), 0.0], [0.0, np.log(2)]],
)
parameter_values["Negative particle diffusivity [m2.s-1]"] = pybop.Parameter(
    transformation=pybop.LogTransformation(),
    distribution=pybop.MarginalDistribution(distribution, 0),
)
parameter_values["Positive particle diffusivity [m2.s-1]"] = pybop.Parameter(
    transformation=pybop.LogTransformation(),
    distribution=pybop.MarginalDistribution(distribution, 1),
)

# Set up simulator with custom settings
submesh_types, var_pts, spatial_methods = spectral_mesh_pts_and_method(10, 10, 10)
simulator = pybop.pybamm.Simulator(
    model=model,
    parameter_values=parameter_values,
    protocol=pybamm.Experiment(
        [
            "Discharge at 1.0 C for 15 minutes (1 second period)",
            "Rest for 15 minutes (1 second period)",
        ]
    ),
    solver=pybamm.CasadiSolver(
        rtol=1e-5,
        atol=1e-5,
        root_tol=1e-3,
        max_step_decrease_count=10,
        extra_options_setup={
            "disable_internal_warnings": True,
            "newton_scheme": "tfqmr",
        },
        return_solution_if_failed_early=True,
    ),
    output_variables=["Voltage [V]"],
    submesh_types=submesh_types,
    var_pts=var_pts,
    spatial_methods=spatial_methods,
)

# Generate synthetic data
synthetic_data = simulator.solve(
    inputs={
        "Negative particle diffusivity [m2.s-1]": original_D_n,
        "Positive particle diffusivity [m2.s-1]": original_D_p,
    }
)
dataset = pybop.import_pybamm_solution(synthetic_data)

ICI_cost = pybop.SquareRootFeatureDistance(
    dataset=dataset,
    target="Voltage [V]",
    feature="inverse_slope",
    time_start=0,
    time_end=90,
)
GITT_cost = pybop.SquareRootFeatureDistance(
    dataset=dataset,
    target="Voltage [V]",
    feature="inverse_slope",
    time_start=901,
    time_end=991,
)

if __name__ == "__main__":
    ICI_problem = pybop.Problem(simulator, ICI_cost)
    GITT_problem = pybop.Problem(simulator, GITT_cost)
    problem = pybop.MetaProblem(ICI_problem, GITT_problem)

    # Set up and run the optimiser, increase the number of iterations
    # and samples to improve accuracy
    options = pybop.EPBOLFIOptions(
        ep_iterations=2,
        ep_total_dampening=0,
        bolfi_initial_sobol_samples=8,
        bolfi_optimally_acquired_samples=8,
        bolfi_posterior_effective_sample_size=8,
        posterior_gelman_rubin_threshold=1.2,
        verbose=True,
        model_parameter_boundaries={
            "Negative particle diffusivity [m2.s-1]": [
                original_D_n / 2,
                original_D_n * 2,
            ],
            "Positive particle diffusivity [m2.s-1]": [
                original_D_p / 2,
                original_D_p * 2,
            ],
        },
    )
    optim = pybop.EP_BOLFI(problem, options=options)
    result = optim.run()
    print("True values:", [original_D_n, original_D_p])

    # Plot the optimisation result
    result.plot_convergence(yaxis={"type": "log"})
    result.plot_parameters(yaxis={"type": "log"}, yaxis2={"type": "log"})

    # Plot the prior and posterior distributions
    pybop.plot.distribution(result.problem.parameters, result.posterior)

    # Plot predictions for a set of inputs sampled from the posterior
    fig = result.plot_predictive(show=False)
    fig[0].show()

    pybamm.print_citations()
