import numpy as np
import pytest
from pybamm import Parameter

import pybop
from pybop._classification import classify_using_Hessian


class TestClassification:
    """
    A class to test the classification of different optimisation results.
    """

    @pytest.fixture(
        params=[
            np.asarray([0.05, 0.05]),
            np.asarray([0.1, 0.05]),
            np.asarray([0.05, 0.01]),
        ]
    )
    def parameters(self, request):
        ground_truth = request.param
        return pybop.Parameters(
            pybop.Parameter(
                "R0 [Ohm]",
                prior=pybop.Gaussian(0.05, 0.01),
                bounds=[0.02, 0.08],
                true_value=ground_truth[0],
            ),
            pybop.Parameter(
                "R1 [Ohm]",
                prior=pybop.Gaussian(0.05, 0.01),
                bounds=[0.02, 0.08],
                true_value=ground_truth[1],
            ),
        )

    @pytest.fixture
    def parameter_set(self):
        parameter_set = pybop.ParameterSet(
            json_path="examples/parameters/initial_ecm_parameters.json"
        )
        parameter_set.import_parameters()
        parameter_set.params.update({"C1 [F]": 1000})
        return parameter_set

    @pytest.fixture
    def model(self, parameter_set, parameters):
        parameter_set.params.update(parameters.as_dict(parameters.true_value()))
        return pybop.empirical.Thevenin(parameter_set=parameter_set)

    @pytest.fixture
    def dataset(self, model):
        experiment = pybop.Experiment(
            [
                (
                    "Discharge at 0.5C for 2 minutes (4 seconds period)",
                    "Charge at 0.5C for 2 minutes (4 seconds period)",
                ),
            ]
        )
        solution = model.predict(experiment=experiment)
        return pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data,
            }
        )

    @pytest.fixture
    def problem(self, model, parameters, dataset):
        return pybop.FittingProblem(model, parameters, dataset)

    @pytest.mark.integration
    def test_classify_using_Hessian(self, problem):
        cost = pybop.RootMeanSquaredError(problem)
        optim = pybop.XNES(cost=cost)

        x = np.asarray(cost.parameters.true_value())
        bounds = cost.parameters.get_bounds()
        optim_x = np.clip(x, bounds["lower"], bounds["upper"])

        if np.all(x == np.asarray([0.05, 0.05])):
            message = classify_using_Hessian(optim, optim_x)
            assert message == "The optimiser has located a minimum."
        elif np.all(x == np.asarray([0.1, 0.05])):
            message = classify_using_Hessian(optim, optim_x)
            assert message == (
                "The optimiser has not converged to a stationary point."
                " The result is near the upper bound of R0 [Ohm]."
            )
        elif np.all(x == np.asarray([0.05, 0.01])):
            message = classify_using_Hessian(optim, optim_x)
            assert message == (
                "The optimiser has not converged to a stationary point."
                " The result is near the lower bound of R1 [Ohm]."
            )
        else:
            raise Exception("Please add a check for these values.")

        if np.all(x == np.asarray([0.05, 0.05])):
            cost = pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=0.002)
            optim = pybop.XNES(cost=cost)

            message = classify_using_Hessian(optim, x)
            assert message == "The optimiser has located a maximum."

        # message = classify_using_Hessian(optim, x)
        # assert message == "The optimiser has located a saddle point."

    @pytest.mark.integration
    def test_insensitive_classify_using_Hessian(self, parameter_set):
        param_R0 = pybop.Parameter(
            "R0 [Ohm]",
            bounds=[0, 0.002],
            true_value=0.001,
        )
        param_R0_mod = pybop.Parameter(
            "R0 modification [Ohm]",
            bounds=[-1e-4, 1e-4],
            true_value=0,
        )
        param_R1_mod = pybop.Parameter(
            "R1 modification [Ohm]",
            bounds=[-1e-4, 1e-4],
            true_value=0,
        )
        parameter_set.params.update(
            {
                "R0 modification [Ohm]": 0,
                "R1 modification [Ohm]": 0,
            },
            check_already_exists=False,
        )
        R0, R1 = parameter_set["R0 [Ohm]"], parameter_set["R1 [Ohm]"]
        parameter_set.params.update(
            {
                "R0 [Ohm]": R0 + Parameter("R0 modification [Ohm]"),
                "R1 [Ohm]": R1 + Parameter("R1 modification [Ohm]"),
            }
        )
        model = pybop.empirical.Thevenin(parameter_set=parameter_set)

        experiment = pybop.Experiment(
            ["Discharge at 0.5C for 2 minutes (4 seconds period)"]
        )
        solution = model.predict(experiment=experiment)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data,
            }
        )

        for i, parameters in enumerate(
            [
                pybop.Parameters(param_R0_mod, param_R1_mod),
                pybop.Parameters(param_R1_mod, param_R0),
                pybop.Parameters(param_R0, param_R1_mod),
            ]
        ):
            problem = pybop.FittingProblem(model, parameters, dataset)
            cost = pybop.SumofPower(problem, p=3)
            optim = pybop.XNES(cost=cost)

            x = np.asarray(cost.parameters.true_value())
            message = classify_using_Hessian(optim, x)

            if i == 0:
                assert message == "The cost is insensitive to these parameters."
            elif i == 1 or i == 2:
                assert message == "The cost is insensitive to R1 modification [Ohm]."

        parameters = pybop.Parameters(param_R0, param_R0_mod)
        problem = pybop.FittingProblem(model, parameters, dataset)
        cost = pybop.SumofPower(problem, p=3)
        optim = pybop.XNES(cost=cost)

        x = np.asarray(cost.parameters.true_value())
        message = classify_using_Hessian(optim, x)
        assert message == "There may be a correlation between these parameters."

    @pytest.mark.integration
    def test_classify_using_Hessian_invalid(self, model, parameters, dataset):
        parameters.remove("R0 [Ohm]")
        problem = pybop.FittingProblem(model, parameters, dataset)
        cost = pybop.SumSquaredError(problem)
        optim = pybop.SNES(cost=cost)
        optim.run()

        with pytest.raises(
            ValueError,
            match="The function classify_using_Hessian currently only works"
            " in the case of 2 parameters.",
        ):
            classify_using_Hessian(optim)
