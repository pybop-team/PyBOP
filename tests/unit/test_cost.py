from copy import copy

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

    def test_base(self, model_and_parameter_values, parameters, dataset):
        model, parameter_values = model_and_parameter_values
        simulator = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            input_parameter_names=parameters.names,
            protocol=dataset,
        )
        problem = pybop.FittingProblem(simulator, parameters, dataset)
        for cost_class in [pybop.BaseCost, pybop.FittingCost, pybop.BaseLikelihood]:
            base_cost = cost_class(problem)
            assert base_cost.problem == problem
            with pytest.raises(NotImplementedError):
                base_cost([0.5])

    @pytest.fixture(params=[2.5, 3.777])
    def problem(self, model_and_parameter_values, parameters, dataset, request):
        cut_off = request.param
        model, parameter_values = model_and_parameter_values
        parameter_values.update({"Lower voltage cut-off [V]": cut_off})
        simulator = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            input_parameter_names=parameters.names,
            protocol=dataset,
        )
        return pybop.FittingProblem(simulator, parameters, dataset)

    @pytest.fixture(
        params=[
            pybop.MeanAbsoluteError,
            pybop.LogPosterior,
        ]
    )
    def fitting_cost(self, problem, request):
        cls = request.param
        if cls is pybop.MeanAbsoluteError:
            return cls(problem)
        elif cls is pybop.LogPosterior:
            return cls(pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=0.002))

    def test_fitting_costs(self, fitting_cost, parameters):
        cost = fitting_cost

        # Test cost direction
        if isinstance(cost, pybop.BaseLikelihood):
            higher_cost = cost([0.52])
            lower_cost = cost([0.55])
        else:
            higher_cost = cost([0.55])
            lower_cost = cost([0.52])
        assert higher_cost > lower_cost or (
            higher_cost == lower_cost and not np.isfinite(higher_cost)
        )

        # Test type of returned value
        assert np.isscalar(cost([0.5]))
        assert np.isscalar(cost(parameters.to_dict()))

        if isinstance(cost, pybop.MeanAbsoluteError):
            assert cost([0.5]) >= 0

            # Test option setting
            cost.set_fail_gradient(10)
            assert cost._de == 10

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
    def test_error_measures(self, problem, cost_class, expected_name):
        cost = cost_class(problem)
        assert cost.name == expected_name
        assert cost.problem.has_sensitivities

        # Test cost direction
        higher_cost = cost([0.55])
        lower_cost = cost([0.52])
        assert higher_cost > lower_cost or (
            higher_cost == lower_cost and not np.isfinite(higher_cost)
        )

        e, de = cost([0.5], calculate_grad=True)

        assert np.isscalar(e)
        assert isinstance(de, np.ndarray)

    def test_minkowski(self, problem):
        # Incorrect order
        with pytest.raises(ValueError, match="The order of the Minkowski distance"):
            pybop.Minkowski(problem, p=-1)
        with pytest.raises(
            ValueError,
            match="For p = infinity, an implementation of the Chebyshev distance is required.",
        ):
            pybop.Minkowski(problem, p=np.inf)

    def test_sumofpower(self, problem):
        # Incorrect order
        with pytest.raises(
            ValueError, match="The order of 'p' must be greater than 0."
        ):
            pybop.SumOfPower(problem, p=-1)

        with pytest.raises(ValueError, match="p = np.inf is not yet supported."):
            pybop.SumOfPower(problem, p=np.inf)

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
            input_parameter_names=parameters.names,
            protocol=dataset,
        )
        problem = pybop.FittingProblem(simulator, parameters, dataset)
        cost = cost_class(problem, weighting=1.0)
        x = [0.5]
        e, de = cost(x, calculate_grad=True)

        # Test that the equal weighting is the same as weighting by one
        costE = cost_class(problem, weighting="equal")
        eE, deE = costE(x, calculate_grad=True)
        np.testing.assert_allclose(e, eE)
        np.testing.assert_allclose(de, deE)

        # Test that domain-based weighting also matches for evenly spaced data
        costD = cost_class(problem, weighting="domain")
        eD, deD = costD(x, calculate_grad=True)
        np.testing.assert_allclose(e, eD)
        np.testing.assert_allclose(de, deD)

        # Test that the domain-based weighting accounts for random spacing in the dataset
        simulator = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            input_parameter_names=parameters.names,
            protocol=randomly_spaced_dataset,
        )
        problemR = pybop.FittingProblem(simulator, parameters, randomly_spaced_dataset)
        costR = cost_class(problemR, weighting="domain")
        eR, deR = costR(x, calculate_grad=True)
        np.testing.assert_allclose(e, eR, rtol=1e-2, atol=1e-9)
        np.testing.assert_allclose(de, deR, rtol=1e-2, atol=1e-9)

        # Check that the sum (and therefore mean) are the same as an even weighting
        np.testing.assert_allclose(np.sum(costR.weighting), len(costR.weighting))

        # Check gradient calculation using finite difference
        delta = 1e-6 * x[0]
        cost_right = costR([x[0] + delta / 2])
        cost_left = costR([x[0] - delta / 2])
        numerical_grad = (cost_right - cost_left) / delta
        np.testing.assert_allclose(deR, numerical_grad, rtol=6e-3)

    @pytest.fixture
    def design_problem(self, parameters, experiment):
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

        simulator = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            input_parameter_names=parameters.names,
            protocol=experiment,
            initial_state={"Initial SoC": 0.5},
        )
        return pybop.DesignProblem(
            simulator,
            parameters,
            output_variables=[
                "Gravimetric energy density [Wh.kg-1]",
                "Volumetric energy density [Wh.m-3]",
                "Gravimetric power density [W.kg-1]",
                "Volumetric power density [W.m-3]",
            ],
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
    def test_design_costs(self, target, design_problem):
        cost = pybop.DesignCost(design_problem, target=target)

        # Test type of returned value
        assert np.isscalar(cost([0.5]))
        assert cost([0.4]) >= 0  # Should be a viable design
        assert cost([0.8]) == -np.inf  # Should exceed active material + porosity < 1
        assert cost([1.4]) == -np.inf  # Definitely not viable
        assert cost([-0.1]) == -np.inf  # Should not be a viable design

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
        simulator = pybop.pybamm.Simulator(
            model,
            parameter_values=parameter_values,
            input_parameter_names=parameters.names,
            protocol=noisy_dataset,
        )
        return pybop.FittingProblem(simulator, parameters, noisy_dataset)

    def test_weighted_fitting_cost(self, noisy_problem):
        problem = noisy_problem
        cost1 = pybop.SumSquaredError(problem)
        cost2 = pybop.RootMeanSquaredError(problem)

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
            pybop.WeightedCost(cost1.problem)
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
        assert weighted_cost_2.has_identical_problems is True
        assert weighted_cost_2.has_separable_problem is False
        assert weighted_cost_2.problem is problem
        assert weighted_cost_2([0.5]) >= 0
        np.testing.assert_allclose(
            weighted_cost_2([0.6]),
            cost1([0.6]) + weight * cost2([0.6]),
            atol=1e-5,
        )

        # Test with different problems
        cost3 = pybop.RootMeanSquaredError(copy(problem))
        weighted_cost_3 = pybop.WeightedCost(cost1, cost3, weights=[1, weight])
        assert weighted_cost_3.has_identical_problems is False
        assert weighted_cost_3.has_separable_problem is False
        assert weighted_cost_3.problem is None
        assert weighted_cost_3([0.5]) >= 0
        np.testing.assert_allclose(
            weighted_cost_3([0.6]),
            cost1([0.6]) + weight * cost3([0.6]),
            atol=1e-5,
        )

        errors_2, sensitivities_2 = weighted_cost_2([0.5], calculate_grad=True)
        errors_3, sensitivities_3 = weighted_cost_3([0.5], calculate_grad=True)
        np.testing.assert_allclose(errors_2, errors_3, atol=1e-5)
        np.testing.assert_allclose(sensitivities_2, sensitivities_3, atol=1e-5)

        # Test LogPosterior explicitly
        cost4 = pybop.LogPosterior(pybop.GaussianLogLikelihood(problem))
        weighted_cost_4 = pybop.WeightedCost(cost1, cost4, weights=[1, 1 / weight])
        assert weighted_cost_4.has_identical_problems is True
        assert weighted_cost_4.has_separable_problem is False
        sigma = 0.01
        assert np.isfinite(cost4.parameters["Sigma for output 1"].prior.logpdf(sigma))
        assert np.isfinite(weighted_cost_4([0.5, sigma]))
        np.testing.assert_allclose(
            weighted_cost_4([0.6, sigma]),
            cost1([0.6]) - 1 / weight * cost4([0.6, sigma]),
            atol=1e-5,
        )

    def test_weighted_design_cost(self, design_problem):
        cost1 = pybop.DesignCost(
            design_problem, target="Gravimetric energy density [Wh.kg-1]"
        )
        cost2 = pybop.DesignCost(
            design_problem, target="Volumetric energy density [Wh.m-3]"
        )

        # Test DesignCosts with identical problems
        weighted_cost = pybop.WeightedCost(cost1, cost2)
        assert weighted_cost.has_identical_problems is True
        assert weighted_cost.has_separable_problem is False
        assert weighted_cost.problem is design_problem
        assert weighted_cost([0.5]) >= 0
        np.testing.assert_allclose(
            weighted_cost([0.6]),
            cost1([0.6]) + cost2([0.6]),
            atol=1e-5,
        )

        # Test DesignCosts with different problems
        cost3 = pybop.DesignCost(
            copy(design_problem), target="Volumetric energy density [Wh.m-3]"
        )
        weighted_cost = pybop.WeightedCost(cost1, cost3)
        assert weighted_cost.has_identical_problems is False
        assert weighted_cost.has_separable_problem is False
        for i, _ in enumerate(weighted_cost.costs):
            assert isinstance(weighted_cost.costs[i].problem, pybop.DesignProblem)

        assert weighted_cost([0.5]) >= 0
        np.testing.assert_allclose(
            weighted_cost([0.6]),
            cost1([0.6]) + cost2([0.6]),
            atol=1e-5,
        )

    def test_mixed_problem_classes(self, problem, design_problem):
        cost1 = pybop.SumSquaredError(problem)
        cost2 = pybop.DesignCost(
            design_problem, target="Gravimetric energy density [Wh.kg-1]"
        )
        with pytest.raises(
            TypeError,
            match="All problems must be of the same class type.",
        ):
            pybop.WeightedCost(cost1, cost2)

    def test_parameter_sensitivities(self, problem):
        cost = pybop.MeanAbsoluteError(problem)
        result = cost.sensitivity_analysis(4)

        # Assertions
        assert isinstance(result, dict)
        assert "S1" in result
        assert "ST" in result
        assert isinstance(result["S1"], np.ndarray)
        assert isinstance(result["S2"], np.ndarray)
        assert isinstance(result["ST"], np.ndarray)
        assert isinstance(result["S1_conf"], np.ndarray)
        assert isinstance(result["ST_conf"], np.ndarray)
        assert isinstance(result["S2_conf"], np.ndarray)
        assert result["S1"].shape == (1,)
        assert result["ST"].shape == (1,)
