import argparse
import os

import pyro
from pyro.infer import MCMC, NUTS
import pyro.distributions as dist

import pybamm
import numpy as np

pyro.set_rng_seed(0)


def solution(t_eval, y, model, inputs):
    termination = "final time"
    t_event = None
    y_event = np.array(None)

    intern_sol = pybamm.Solution(
        t_eval, y, model, inputs, t_event, y_event, termination
    )
    return intern_sol["Voltage [V]"].data


def model(s, t_eval, m):
    theta0 = pyro.sample(
        "theta0", dist.TruncatedNormal(loc=0.3, scale=0.1, low=0.2, high=0.5)
    )
    theta1 = pyro.sample(
        "theta1", dist.TruncatedNormal(loc=0.5, scale=0.1, low=0.4, high=0.6)
    )

    inputs = {
        "Negative electrode active material volume fraction": theta0,
        "Positive electrode active material volume fraction": theta1,
    }

    # Attempt 1
    V = s.solve(m, t_eval, inputs=inputs)["Voltage [V]"].data

    # Attempt 2
    # V = pure_callback(
    #     solution, jnp.zeros(100), t_eval=t_eval, y=yhat, model=m, inputs=inputs
    # )


def main(args):
    # load model
    m = pybamm.lithium_ion.SPMe()
    m.convert_to_format = "jax"
    m.events = []

    # create geometry
    geometry = m.default_geometry

    # load parameter values and process model and geometry
    param = m.default_parameter_values
    param.update(
        {
            "Negative electrode active material volume fraction": "[input]",
            "Positive electrode active material volume fraction": "[input]",
        }
    )
    param.process_geometry(geometry)
    param.process_model(m)

    # set mesh
    mesh = pybamm.Mesh(geometry, m.default_submesh_types, m.default_var_pts)

    # discretise model
    disc = pybamm.Discretisation(mesh, m.default_spatial_methods)
    disc.process_model(m)

    # initial solve to construct solver class
    t_eval = np.linspace(0, 3600, 100)
    s = pybamm.CasadiSolver(atol=1e-6, rtol=1e-6)
    s.solve(
        m,
        t_eval,
        inputs={
            "Negative electrode active material volume fraction": 0.3,
            "Positive electrode active material volume fraction": 0.5,
        },
    )

    # run numpyro inference
    nuts_kernel = NUTS(model, jit_compile=args.jit)
    mcmc = MCMC(
        nuts_kernel,
        num_samples=args.num_samples,
        warmup_steps=args.num_warmup,
        num_chains=args.num_chains,
        # progress_bar=True,
    )
    mcmc.run(model, t_eval=t_eval, m=m)
    mcmc.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyBaMM Model")
    parser.add_argument("--num-samples", nargs="?", default=100, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=20, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--jit", action="store_true", default=False)
    args = parser.parse_args()

    # pyro.set_platform(args.device)
    # pyro.set_host_device_count(args.num_chains)

    main(args)
