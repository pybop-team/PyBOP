import numpy as np
import pybamm
import pytest

import pybop


class TestCosts:
    """
    Class for tests cost functions
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def model_and_parameter_values(self, ground_truth):
        model = pybamm.lithium_ion.SPM()
        parameter_values = model.default_parameter_values
        parameter_values["Negative electrode active material volume fraction"] = (
            ground_truth
        )
        return model, parameter_values

    @pytest.fixture
    def ground_truth(self):
        return 0.52

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.01),
                bounds=[0.375, 0.625],
            )
        )

    @pytest.fixture
    def experiment(self):
        return pybamm.Experiment(["Discharge at 1C for 10 minutes (20 second period)"])

    @pytest.fixture
    def dataset(self, model_and_parameter_values, experiment):
        model, parameter_values = model_and_parameter_values
        solution = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        ).solve()
        return pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Terminal voltage [V]"].data,
            }
        )

    def test_base(self, dataset):
        cost = pybop.ErrorMeasure(dataset)
        with pytest.raises(NotImplementedError):
            cost([0.5])

    @pytest.fixture(params=[2.5, 3.777])
    def simulator(self, model_and_parameter_values, parameters, dataset, request):
        cut_off = request.param
        model, parameter_values = model_and_parameter_values
        parameter_values.update({"Lower voltage cut-off [V]": cut_off})
        return pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            parameters=parameters,
            protocol=dataset,
        )

    @pytest.mark.parametrize(
        "cost_class",
        [
            pybop.MeanAbsoluteError,
            pybop.GaussianLogLikelihoodKnownSigma,
            pybop.LogPosterior,
        ],
    )
    def test_fitting_costs(self, simulator, parameters, dataset, cost_class):
        if cost_class is pybop.LogPosterior:
            likelihood = pybop.GaussianLogLikelihoodKnownSigma(dataset, sigma0=0.002)
            cost = cost_class(likelihood)
        elif issubclass(cost_class, pybop.LogLikelihood):
            cost = cost_class(dataset, sigma0=0.002)
        else:
            cost = cost_class(dataset)
        problem = pybop.Problem(simulator, cost)

        # Test cost direction
        if isinstance(cost, pybop.LogLikelihood):
            higher_cost = cost([0.52])
            lower_cost = cost([0.55])
        else:
            higher_cost = cost([0.55])
            lower_cost = cost([0.52])
        assert higher_cost > lower_cost or (
            higher_cost == lower_cost and not np.isfinite(higher_cost)
        )

        # Test type of returned value
        assert np.isscalar(problem([0.5]))
        assert np.isscalar(problem(parameters.to_dict()))

        if isinstance(cost, pybop.MeanAbsoluteError):
            assert problem([0.5]) >= 0

            # Test option setting
            problem._cost.set_fail_gradient(10)
            assert problem._cost._de == 10

    @pytest.mark.parametrize(
        "cost_class, expected_name",
        [
            (pybop.MeanSquaredError, "Mean Squared Error"),
            (pybop.RootMeanSquaredError, "Root Mean Squared Error"),
            (pybop.MeanAbsoluteError, "Mean Absolute Error"),
            (pybop.SumSquaredError, "Sum Squared Error"),
            (pybop.Minkowski, "Minkowski"),
            (pybop.SumOfPower, "Sum Of Power"),
        ],
    )
    def test_error_measures(
        self, simulator, parameters, dataset, cost_class, expected_name
    ):
        cost = cost_class(dataset)
        assert cost.name == expected_name

        problem = pybop.Problem(simulator, cost)

        # Test cost direction
        higher_cost = problem([0.55])
        lower_cost = problem([0.52])
        assert higher_cost > lower_cost or (
            higher_cost == lower_cost and not np.isfinite(higher_cost)
        )

        e, de = problem([0.5], calculate_grad=True)

        assert np.isscalar(e)
        assert isinstance(de, np.ndarray)

    def test_minkowski(self, dataset):
        # Incorrect order
        with pytest.raises(ValueError, match="The order of the Minkowski distance"):
            pybop.Minkowski(dataset, p=-1)
        with pytest.raises(
            ValueError,
            match="For p = infinity, an implementation of the Chebyshev distance is required.",
        ):
            pybop.Minkowski(dataset, p=np.inf)

    def test_sumofpower(self, dataset):
        # Incorrect order
        with pytest.raises(
            ValueError, match="The order of 'p' must be greater than 0."
        ):
            pybop.SumOfPower(dataset, p=-1)

        with pytest.raises(ValueError, match="p = np.inf is not yet supported."):
            pybop.SumOfPower(dataset, p=np.inf)

    @pytest.fixture
    def randomly_spaced_dataset(self, model_and_parameter_values):
        model, parameter_values = model_and_parameter_values
        t_eval = np.linspace(0, 10 * 60, 31) + np.concatenate(
            ([0], np.random.normal(0, 1, 29), [0])
        )
        solution = pybamm.Simulation(model, parameter_values=parameter_values).solve(
            t_eval=t_eval,
            t_interp=t_eval,
        )
        return pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Terminal voltage [V]"].data,
            }
        )

    @pytest.mark.parametrize(
        "cost_class",
        [
            pybop.MeanSquaredError,
            pybop.RootMeanSquaredError,
            pybop.MeanAbsoluteError,
            pybop.SumSquaredError,
            pybop.Minkowski,
            pybop.SumOfPower,
        ],
    )
    def test_error_weighting(
        self,
        model_and_parameter_values,
        parameters,
        dataset,
        randomly_spaced_dataset,
        cost_class,
    ):
        model, parameter_values = model_and_parameter_values
        simulator = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            parameters=parameters,
            protocol=dataset,
        )
        cost = cost_class(dataset, weighting=1.0)
        problem = pybop.Problem(simulator, cost)
        x = [0.5]
        e, de = problem(x, calculate_grad=True)

        # Test that the equal weighting is the same as weighting by one
        costE = cost_class(dataset, weighting="equal")
        problemE = pybop.Problem(simulator, costE)
        eE, deE = problemE(x, calculate_grad=True)
        np.testing.assert_allclose(e, eE)
        np.testing.assert_allclose(de, deE)

        # Test that domain-based weighting also matches for evenly spaced data
        costD = cost_class(dataset, weighting="domain")
        problemD = pybop.Problem(simulator, costD)
        eD, deD = problemD(x, calculate_grad=True)
        np.testing.assert_allclose(e, eD)
        np.testing.assert_allclose(de, deD)

        # Test that the domain-based weighting accounts for random spacing in the dataset
        simulator = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            parameters=parameters,
            protocol=randomly_spaced_dataset,
        )
        costR = cost_class(randomly_spaced_dataset, weighting="domain")
        problemR = pybop.Problem(simulator, costR)
        eR, deR = problemR(x, calculate_grad=True)
        np.testing.assert_allclose(e, eR, rtol=1e-2, atol=1e-9)
        np.testing.assert_allclose(de, deR, rtol=1e-2, atol=1e-9)

        # Check that the sum (and therefore mean) are the same as an even weighting
        np.testing.assert_allclose(
            np.sum(problemR.cost.weighting),
            len(problemR.cost.weighting),
        )

        # Check gradient calculation using finite difference
        delta = 1e-6 * x[0]
        cost_right = problemR([x[0] + delta / 2])
        cost_left = problemR([x[0] - delta / 2])
        numerical_grad = (cost_right - cost_left) / delta
        np.testing.assert_allclose(deR, numerical_grad, rtol=6e-3)

    @pytest.fixture
    def design_simulator(self, parameters, experiment):
        model = pybamm.lithium_ion.SPM()
        target_time = 600  # length of dis/charge in the experiment [s]
        pybop.pybamm.add_variable_to_model(
            model, "Gravimetric energy density [Wh.kg-1]"
        )
        pybop.pybamm.add_variable_to_model(model, "Volumetric energy density [Wh.m-3]")
        pybop.pybamm.add_variable_to_model(
            model, "Gravimetric power density [W.kg-1]", target_time=target_time
        )
        pybop.pybamm.add_variable_to_model(
            model, "Volumetric power density [W.m-3]", target_time=target_time
        )

        parameter_values = pybamm.ParameterValues("Chen2020")
        pybop.pybamm.set_formation_concentrations(parameter_values)
        parameter_values.update(
            {
                "Electrolyte density [kg.m-3]": pybamm.Parameter(
                    "Separator density [kg.m-3]"
                ),
                "Negative electrode active material density [kg.m-3]": pybamm.Parameter(
                    "Negative electrode density [kg.m-3]"
                ),
                "Negative electrode carbon-binder density [kg.m-3]": pybamm.Parameter(
                    "Negative electrode density [kg.m-3]"
                ),
                "Positive electrode active material density [kg.m-3]": pybamm.Parameter(
                    "Positive electrode density [kg.m-3]"
                ),
                "Positive electrode carbon-binder density [kg.m-3]": pybamm.Parameter(
                    "Positive electrode density [kg.m-3]"
                ),
                "Cell mass [kg]": pybop.pybamm.cell_mass(),
                "Cell volume [m3]": pybop.pybamm.cell_volume(),
            },
            check_already_exists=False,
        )

        return pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            parameters=parameters,
            protocol=experiment,
            initial_state={"Initial SoC": 0.5},
        )

    @pytest.mark.parametrize(
        "target",
        [
            "Gravimetric energy density [Wh.kg-1]",
            "Volumetric energy density [Wh.m-3]",
            "Gravimetric power density [W.kg-1]",
            "Volumetric power density [W.m-3]",
        ],
    )
    def test_design_costs(self, target, design_simulator, parameters):
        cost = pybop.DesignCost(target=target)
        design_problem = pybop.Problem(design_simulator, cost)

        # Test type of returned value
        assert np.isscalar(design_problem([0.5]))
        assert design_problem([0.4]) >= 0  # Should be a viable design
        assert (
            design_problem([0.8]) == -np.inf
        )  # Should exceed active material + porosity < 1
        assert design_problem([1.4]) == -np.inf  # Definitely not viable
        assert design_problem([-0.1]) == -np.inf  # Should not be a viable design

    @pytest.fixture
    def noisy_problem(self, ground_truth, parameters, experiment):
        model = pybamm.lithium_ion.SPM()
        parameter_values = model.default_parameter_values
        parameter_values["Negative electrode active material volume fraction"] = (
            ground_truth
        )
        solution = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        ).solve()
        noisy_dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data
                + np.random.normal(0, 0.02, len(solution["Time [s]"].data)),
            }
        )
        return noisy_dataset, pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            parameters=parameters,
            protocol=noisy_dataset,
        )

    def test_weighted_fitting_cost(self, noisy_problem, parameters, dataset):
        dataset, simulator = noisy_problem
        cost1 = pybop.SumSquaredError(dataset)
        cost2 = pybop.RootMeanSquaredError(dataset)

        # Test with and without weights
        weighted_cost = pybop.WeightedCost(cost1, cost2)
        np.testing.assert_array_equal(weighted_cost.weights, np.ones(2))
        weighted_cost = pybop.WeightedCost(cost1, cost2, weights=[1, 1])
        np.testing.assert_array_equal(weighted_cost.weights, np.ones(2))
        weighted_cost = pybop.WeightedCost(cost1, cost2, weights=np.array([1, 1]))
        np.testing.assert_array_equal(weighted_cost.weights, np.ones(2))
        with pytest.raises(
            TypeError,
            match="All costs must be instances of BaseCost.",
        ):
            pybop.WeightedCost(parameters)
        with pytest.raises(
            ValueError,
            match="Weights must be numeric values.",
        ):
            pybop.WeightedCost(cost1, cost2, weights="Invalid string")
        with pytest.raises(
            ValueError,
            match="Number of weights must match number of costs.",
        ):
            pybop.WeightedCost(cost1, cost2, weights=[1])

        # Test with identical problems
        weight = 100
        weighted_cost_2 = pybop.WeightedCost(cost1, cost2, weights=[1, weight])
        problem_1 = pybop.Problem(simulator, cost1)
        problem_2 = pybop.Problem(simulator, cost2)
        weighted_2 = pybop.Problem(simulator, weighted_cost_2)
        assert weighted_2([0.5]) >= 0
        np.testing.assert_allclose(
            weighted_2([0.6]),
            problem_1([0.6]) + weight * problem_2([0.6]),
            atol=1e-5,
        )

        # Test with different problems
        cost3 = pybop.RootMeanSquaredError(dataset)
        weighted_cost_3 = pybop.WeightedCost(cost1, cost3, weights=[1, weight])
        problem_3 = pybop.Problem(simulator, cost3)
        weighted_3 = pybop.Problem(simulator, weighted_cost_3)
        assert weighted_3([0.5]) >= 0
        np.testing.assert_allclose(
            weighted_3([0.6]),
            problem_1([0.6]) + weight * problem_3([0.6]),
            atol=1e-5,
        )

        errors_2, sensitivities_2 = weighted_2([0.5], calculate_grad=True)
        errors_3, sensitivities_3 = weighted_3([0.5], calculate_grad=True)
        np.testing.assert_allclose(errors_2, errors_3, atol=1e-5)
        np.testing.assert_allclose(sensitivities_2, sensitivities_3, atol=1e-5)

        # Test LogPosterior explicitly
        cost4 = pybop.LogPosterior(pybop.GaussianLogLikelihood(dataset))
        weighted_cost_4 = pybop.WeightedCost(cost1, cost4, weights=[1, 1 / weight])
        problem_4 = pybop.Problem(simulator, cost4)
        weighted_4 = pybop.Problem(simulator, weighted_cost_4)
        sigma = 0.01
        assert np.isfinite(cost4.parameters["Sigma for output 1"].prior.logpdf(sigma))
        assert np.isfinite(weighted_4([0.5, sigma]))
        np.testing.assert_allclose(
            weighted_4([0.6, sigma]),
            problem_1([0.6]) - 1 / weight * problem_4([0.6, sigma]),
            atol=1e-5,
        )
        assert np.isfinite(weighted_4([0.5, sigma]))
        np.testing.assert_allclose(
            weighted_4([0.6, sigma]),
            problem_1([0.6]) - 1 / weight * problem_4([0.6, sigma]),
            atol=1e-5,
        )

    def test_weighted_design_cost(self, design_simulator, parameters):
        cost_1 = pybop.DesignCost(target="Gravimetric energy density [Wh.kg-1]")
        cost_2 = pybop.DesignCost(target="Volumetric energy density [Wh.m-3]")
        problem_1 = pybop.Problem(design_simulator, cost_1)
        problem_2 = pybop.Problem(design_simulator, cost_2)

        weighted_cost = pybop.WeightedCost(cost_1, cost_2)
        problem = pybop.Problem(design_simulator, weighted_cost)
        assert problem([0.5]) >= 0
        np.testing.assert_allclose(
            problem([0.6]), problem_1([0.6]) + problem_2([0.6]), atol=1e-5
        )

    def test_mixed_problem_classes(self, dataset, design_simulator):
        cost1 = pybop.SumSquaredError(dataset)
        cost2 = pybop.DesignCost(target="Gravimetric energy density [Wh.kg-1]")
        with pytest.raises(
            TypeError,
            match="Costs must be either all design costs or all error measures",
        ):
            pybop.WeightedCost(cost1, cost2)
