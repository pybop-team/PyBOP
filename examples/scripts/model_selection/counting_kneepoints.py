import importlib.util
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import array, exp, linspace, log, ndarray, sum
from pybamm import print_citations
from scipy.stats import gaussian_kde
from torch import tensor

import pybop
from pybop.optimisers.sober_basq_optimiser import SOBER_BASQ, SOBER_BASQ_Options


class KneepointModel(pybop.BaseSimulator):
    def __init__(self, parameters, t, n_kneepoints=2):
        super().__init__(parameters)
        self.t = t
        self.output_variables = ["Capacity fade"]
        self.n_kneepoints = n_kneepoints

    def one_kneepoint_model(self, parameters):
        first_slope = parameters[0].reshape(-1, 1)
        kneepoint = parameters[1].reshape(-1, 1)
        second_slope = parameters[2].reshape(-1, 1)

        return (
            (1.0 - first_slope * self.t) * (self.t < kneepoint)
            + ((1.0 - first_slope * kneepoint) - second_slope * (self.t - kneepoint))
            * (self.t >= kneepoint)
        ).T

    def two_kneepoints_model(self, parameters):
        first_slope = parameters[0].reshape(-1, 1)
        first_kneepoint = parameters[1].reshape(-1, 1)
        second_slope = parameters[2].reshape(-1, 1)
        second_kneepoint = parameters[3].reshape(-1, 1) + first_kneepoint
        third_slope = parameters[4].reshape(-1, 1)

        return (
            (1.0 - first_slope * self.t) * (self.t < first_kneepoint)
            + (
                (1.0 - first_slope * first_kneepoint)
                - second_slope * (self.t - first_kneepoint)
            )
            * (self.t >= first_kneepoint)
            * (self.t < second_kneepoint)
            + (
                (1.0 - first_slope * first_kneepoint)
                - second_slope * (second_kneepoint - first_kneepoint)
                - third_slope * (self.t - second_kneepoint)
            )
            * (self.t >= second_kneepoint)
        ).T

    def batch_solve(self, inputs, calculate_sensitivities=False):
        inputs_array = tensor(np.asarray([entry for entry in inputs[0].values()]))
        capacity_fade = self(inputs_array)
        sols = []
        for entry, cf in zip(inputs, capacity_fade):
            sol = pybop.Solution(entry)
            sol.set_solution_variable("Capacity fade", cf)
            sols.append(sol)
        return sols

    def __call__(self, parameters):
        if self.n_kneepoints == 1:
            return self.one_kneepoint_model(parameters)
        elif self.n_kneepoints == 2:
            return self.two_kneepoints_model(parameters)
        else:
            raise ValueError("Only one or two kneepoints are implemented.")


def marginalize_pdf(raw_taken_samples, sober_basq, x_eval=None):
    """Calculate the marginal PDF of the kneepoint(s)."""
    kde = gaussian_kde(raw_taken_samples)
    dim = len(raw_taken_samples)
    kneepoint_pdf_x = []
    kneepoint_pdf_y = []
    for i in range(1, dim, 2):
        # Note: the rough order of parameters may be switched around.
        # sober_basq.diag_order keeps track of that order.
        raw_i = sober_basq.diag_order[i]
        kneepoint_pdf_part_x = []
        kneepoint_pdf_part_y = []
        if x_eval is None:
            k_p_p_min = kde.dataset[raw_i].min()
            k_p_p_max = kde.dataset[raw_i].max()
            raw_kneepoint_edges = linspace(k_p_p_min, k_p_p_max, 101)
        else:
            raw_kneepoint_edges = np.array(
                [
                    sober_basq.apply_transform_and_normalize_one_variable(x, i)
                    for x in x_eval
                ]
            )
            k_p_p_min = raw_kneepoint_edges.min()
            k_p_p_max = raw_kneepoint_edges.max()
        for k0, k1 in zip(raw_kneepoint_edges[:-1], raw_kneepoint_edges[1:]):
            lower_bound = [kde.dataset[j].min() for j in range(dim)]
            lower_bound[raw_i] = k0
            upper_bound = [kde.dataset[j].max() for j in range(dim)]
            upper_bound[raw_i] = k1
            kneepoint_pdf_part_x.append(
                sober_basq.denormalize_and_reverse_transform_one_variable(
                    0.5 * (k0 + k1),
                    i,  # this method uses diag_order
                )
            )
            kneepoint_pdf_part_y.append(kde.integrate_box(lower_bound, upper_bound))
        k_norm = sum(kneepoint_pdf_part_y) * (k_p_p_max - k_p_p_min)
        kneepoint_pdf_part_y = array(kneepoint_pdf_part_y) / k_norm
        kneepoint_pdf_x.append(kneepoint_pdf_part_x)
        kneepoint_pdf_y.append(kneepoint_pdf_part_y)
    return kneepoint_pdf_x, kneepoint_pdf_y, kde


"""
class MeanSquaredErrorPyTorch(pybop.costs.error_measures.ErrorMeasure):

    def __call__(self, r: torch.Tensor) -> torch.Tensor:
        e = torch.sum(r**2, axis=1)**0.5
        return e
"""


if __name__ == "__main__":
    data_index = 0

    spec = importlib.util.spec_from_file_location(
        "read_dataset", "../../data/Baumhofer2014/read_dataset.py"
    )
    read_dataset = importlib.util.module_from_spec(spec)
    sys.modules["read_dataset"] = read_dataset
    spec.loader.exec_module(read_dataset)
    measurements = read_dataset.degradation_data

    fig, ax = plt.subplots(figsize=(2.4 * 2**0.5, 2.4), constrained_layout=True)
    for m in measurements:
        ax.plot(m["Time [s]"], m["Capacity fade"])

    # Cast non-standard dtypes into NumPy floats to avoid PyTorch errors.
    t = ndarray.astype(measurements[data_index]["Time [s]"].T[0], np.float64)[1:]
    t[5] = t[4] + 1
    data = ndarray.astype(measurements[data_index]["Capacity fade"].T[0], np.float64)[
        1:
    ]
    dataset = pybop.Dataset({"Time [s]": t, "Capacity fade": data})
    """
    ax.plot(
        t.cpu(),
        kneepoint_model(
            torch.tensor([[0.0002, 1500, 0.0008]]),
            t
        )[0].cpu(),
        color='black',
        lw=2,
        label='model'
    )
    """
    ax.set_xlabel("Full cycles")
    ax.set_ylabel("Relative remaining capacity")
    # ax.legend()
    plt.show()

    for n_kneepoints, mean, bounds, names in zip(
        (1, 2),
        (array([0.0002, 1500, 0.0008]), array([0.0002, 1500, 0.0008, 2000, 0.001])),
        (
            array([[0.00001, 0.001], [200, 2000], [0.00001, 0.01]]),
            array(
                [
                    [0.00001, 0.001],
                    [200, 2000],
                    [0.00001, 0.01],
                    [700, 2500],
                    [0.00001, 0.01],
                ]
            ),
        ),
        (
            [
                "1st degr. rate [Capacity/Cycle]",
                "1st kneepoint [Cycle]",
                "2nd degr. rate [Capacity/Cycle]",
            ],
            [
                "1st degr. rate [Capacity/Cycle]",
                "1st kneepoint [Cycle]",
                "2nd degr. rate [Capacity/Cycle]",
                "2nd kneepoint [Cycle]",
                "3rd degr. rate [Capacity/Cycle]",
            ],
        ),
    ):
        initial_values = exp(0.5 * (log(bounds.T[0]) + log(bounds.T[1])))
        pybop_prior = pybop.MultivariateParameters(
            {
                n: pybop.Parameter(
                    initial_value=i, bounds=b, transformation=pybop.LogTransformation()
                )
                for n, i, b in zip(names, initial_values, bounds)
            },
            distribution=pybop.MultivariateGaussian(mean=mean, bounds=bounds),
        )
        simulator = KneepointModel(pybop_prior, tensor(t), n_kneepoints)
        # Override the forced univariate Parameters
        simulator.parameters = pybop_prior
        cost = pybop.MeanSquaredError(dataset, "Capacity fade")
        cost._target_data = np.asarray([0])
        pybop_problem = pybop.Problem(simulator, cost)
        # Copy the MultivariateParameters to the meta-problem
        pybop_problem.parameters = simulator.parameters
        pybop_options = SOBER_BASQ_Options(
            model_initial_samples=256,
            sober_iterations=12,
            model_samples_per_iteration=64,
            integration_nodes=256,
            batched_input=True,
        )
        sober_basq_wrapper = SOBER_BASQ(pybop_problem, pybop_options)
        pybop_result = sober_basq_wrapper.run()
        kneepoint_pdf_x, kneepoint_pdf_y, kde = marginalize_pdf(
            pybop_result.posterior.distribution.distribution.dataset,
            sober_basq_wrapper.optimiser,
            x_eval=np.linspace(t[1], t[-1], 201),  # zeroth is 0
        )
        # Sample the predictive posterior.
        posterior_resamples = pybop_result.posterior.rvs(64, apply_transform=True)
        posterior_resamples_pdf = pybop_result.posterior.pdf(posterior_resamples)
        simulations = simulator(tensor(posterior_resamples))
        fig, ax = plt.subplots(figsize=(3 * 2**0.5, 3), layout="constrained")
        norm = matplotlib.colors.Normalize(
            posterior_resamples_pdf.min(), posterior_resamples_pdf.max()
        )
        cmap = plt.get_cmap("viridis")
        for pr, pr_pdf, sim in zip(
            posterior_resamples, posterior_resamples_pdf, simulations
        ):
            ax.plot(
                t,
                simulator(torch.atleast_2d(tensor(pr)).T),
                ls=":",
                color=cmap(norm(pr_pdf)),
            )
        degr_plot = ax.plot(t, data, color="black", lw=2, label="degradation data")[0]
        ax_pdf = ax.twinx()
        ax_pdf.plot(kneepoint_pdf_x[0], np.sum(kneepoint_pdf_y, axis=0))
        fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax_pdf,
            label="Posterior PDF from KDE approximation",
        )
        ax.set_xlabel("Cycles")
        ax.set_ylabel("Capacity")
        ax_pdf.set_ylabel("Kneepoints PDF")
        ax.set_ylim((-0.1, 1.1))
        fig.legend(
            [degr_plot],
            [degr_plot.get_label()],
            loc="outside lower center",
        )

        print_citations()

        plt.show()
