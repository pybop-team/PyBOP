import importlib.util
import sys

import matplotlib.pyplot as plt
import numpy as np

# For the machine this was tested on; comment out for default behaviour.
import plotly.io as pio
from pybamm import print_citations
from torch import tensor

import pybop
from pybop.models.kneepoints import KneepointModel
from pybop.optimisers.sober_basq_optimiser import SOBER_BASQ, SOBER_BASQ_Options
from pybop.plot.predictive import predictive

pio.renderers.default = "browser"


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
    t = np.ndarray.astype(measurements[data_index]["Time [s]"].T[0], np.float64)[1:]
    t[5] = t[4] + 1
    data = np.ndarray.astype(
        measurements[data_index]["Capacity fade"].T[0], np.float64
    )[1:]
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
        (
            np.array([0.0002, 1500, 0.0008]),
            np.array([0.0002, 1500, 0.0008, 2000, 0.001]),
        ),
        (
            np.array([[0.00001, 0.001], [200, 2000], [0.00001, 0.01]]),
            np.array(
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
        strict=False,
    ):
        initial_values = np.exp(0.5 * (np.log(bounds.T[0]) + np.log(bounds.T[1])))
        pybop_prior = pybop.MultivariateParameters(
            {
                n: pybop.Parameter(
                    initial_value=i, bounds=b, transformation=pybop.LogTransformation()
                )
                for n, i, b in zip(names, initial_values, bounds, strict=False)
            },
            distribution=pybop.MultivariateGaussian(mean=mean, bounds=bounds),
        )
        simulator = KneepointModel(pybop_prior, tensor(t), n_kneepoints)
        # Override the forced univariate Parameters
        simulator.parameters = pybop_prior
        cost = pybop.MeanSquaredError(dataset, "Capacity fade")
        cost.target_data = np.asarray([0])
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
        kneepoint_pdf_x_eval = np.linspace(np.log(t[1]), np.log(t[-1]), 201)
        kneepoint_pdf_y = np.zeros_like(kneepoint_pdf_x_eval)
        for i in range(n_kneepoints):
            kneepoint_pdf_y += (
                pybop_result.posterior.distribution.distribution.marginal(
                    1 + 2 * i
                ).pdf(kneepoint_pdf_x_eval)
            )
        kneepoint_pdf_x_plot = np.exp(kneepoint_pdf_x_eval)
        fig = predictive(
            pybop_result,
            simulator=lambda p, sim=simulator: (
                sim(tensor(p.T)).T.detach().cpu().numpy()
            ),
            number_of_traces=64,
            dataset_y="Capacity fade",
            data_legend_entry="degradation data",
            rvs_legend_entry="candidate fits",
            pdf_plot=[kneepoint_pdf_x_plot, kneepoint_pdf_y],
            pdf_label="PDF for knee points",
            show=True,
            xaxis_title="Cycles",
            yaxis_title="Capacity",
            yaxis2={"title": "PDF for knee points", "overlaying": "y", "side": "right"},
            yaxis_range=[-0.1, 1.1],
            title=str(n_kneepoints) + "-knee point model",
        )
    print_citations()
