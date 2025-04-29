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
    def model(self, ground_truth):
        model = pybop.lithium_ion.SPM()
        model.parameter_set["Negative electrode active material volume fraction"] = (
            ground_truth
        )
        return model

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
        return pybop.Experiment(["Discharge at 1C for 10 minutes (20 second period)"])

    @pytest.fixture
    def dataset(self, model, experiment):
        solution = model.predict(experiment=experiment)
        return pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Terminal voltage [V]"].data,
            }
        )

    def test_base(self, model, parameters, dataset):
        problem = pybop.FittingProblem(model, parameters, dataset)
        for cost_class in [pybop.BaseCost, pybop.FittingCost, pybop.DesignCost]:
            base_cost = cost_class(problem)
            assert base_cost.problem == problem
            with pytest.raises(NotImplementedError):
                base_cost([0.5])

    @pytest.fixture(params=[2.5, 3.777])
    def problem(self, model, parameters, dataset, request):
        cut_off = request.param
        model.parameter_set.update({"Lower voltage cut-off [V]": cut_off})
        return pybop.FittingProblem(model, parameters, dataset)

    @pytest.fixture(
        params=[
            pybop.MeanAbsoluteError,
            pybop.ObserverCost,
            pybop.LogPosterior,
        ]
    )
    def fitting_cost(self, problem, request):
        cls = request.param
        if cls is pybop.MeanAbsoluteError:
            return cls(problem)
        elif cls is pybop.LogPosterior:
            return cls(pybop.GaussianLogLikelihoodKnownSigma(problem, sigma0=0.002))
        elif cls is pybop.ObserverCost:
            inputs = problem.parameters.initial_value()
            state = problem.model.reinit(inputs)
            n = len(state)
            sigma_diag = [0.0] * n
            sigma_diag[0] = 1e-4
            sigma_diag[1] = 1e-4
            process_diag = [0.0] * n
            process_diag[0] = 1e-4
            process_diag[1] = 1e-4
            sigma0 = np.diag(sigma_diag)
            process = np.diag(process_diag)
            dataset = pybop.Dataset(data_dictionary=problem.dataset)
            return cls(
                pybop.UnscentedKalmanFilterObserver(
                    problem.parameters,
                    problem.model,
                    sigma0=sigma0,
                    process=process,
                    measure=1e-4,
                    dataset=dataset,
                ),
            )

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
        assert np.isscalar(cost(parameters.as_dict()))

        # Test UserWarnings
        if isinstance(cost, (pybop.MeanAbsoluteError)):
            assert cost([0.5]) >= 0
            with pytest.warns(UserWarning):
                cost([1.1])

            # Test option setting
            cost.set_fail_gradient(10)
            assert cost._de == 10

        # Test exception for non-numeric inputs
        with pytest.raises(TypeError, match="Inputs must be a dictionary or numeric."):
            cost(["StringInputShouldNotWork"])

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

        # Test cost direction
        higher_cost = cost([0.55])
        lower_cost = cost([0.52])
        assert higher_cost > lower_cost or (
            higher_cost == lower_cost and not np.isfinite(higher_cost)
        )

        e, de = cost([0.5], calculate_grad=True)

        assert np.isscalar(e)
        assert isinstance(de, np.ndarray)

        # Test exception for non-numeric inputs
        with pytest.raises(TypeError, match="Inputs must be a dictionary or numeric."):
            cost(["StringInputShouldNotWork"], calculate_grad=True)

        with pytest.warns(UserWarning) as record:
            cost([1.1], calculate_grad=True)

        for i in range(len(record)):
            assert "Non-physical point encountered" in str(record[i].message)

        # Test infeasible locations
        cost.problem.model.allow_infeasible_solutions = False
        assert cost([1.1]) == np.inf
        assert cost([1.1], calculate_grad=True) == (np.inf, cost._de)
        assert cost([0.01]) == np.inf
        assert cost([0.01], calculate_grad=True) == (np.inf, cost._de)

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
    def randomly_spaced_dataset(self, model):
        t_eval = np.linspace(0, 10 * 60, 31) + np.concatenate(
            ([0], np.random.normal(0, 1, 29), [0])
        )
        solution = model.predict(t_eval=t_eval)
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
        self, model, parameters, dataset, randomly_spaced_dataset, cost_class
    ):
        problem = pybop.FittingProblem(model, parameters, dataset)
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
        problemR = pybop.FittingProblem(model, parameters, randomly_spaced_dataset)
        costR = cost_class(problemR, weighting="domain")
        eR, deR = costR(x, calculate_grad=True)
        np.testing.assert_allclose(e, eR, rtol=1e-2, atol=1e-9)
        np.testing.assert_allclose(de, deR, rtol=1e-2, atol=1e-9)

        # Check that the sum (and therefore mean) are the same as an even weighting
        np.testing.assert_allclose(np.sum(costR.weighting), len(costR.weighting))

        # Check gradient calculation using finite difference
        delta = 1e-6 * x[0]
        cost_right = costR(x[0] + delta / 2)
        cost_left = costR(x[0] - delta / 2)
        numerical_grad = (cost_right - cost_left) / delta
        np.testing.assert_allclose(deR, numerical_grad, rtol=6e-3)

    @pytest.fixture
    def design_problem(self, parameters, experiment):
        parameter_set = pybop.ParameterSet.pybamm("Chen2020")
        parameter_set.update(
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
            },
            check_already_exists=False,
        )
        model = pybop.lithium_ion.SPM(parameter_set=parameter_set)
        return pybop.DesignProblem(
            model,
            parameters,
            experiment,
            initial_state={"Initial SoC": 0.5},
        )

    @pytest.mark.parametrize(
        "cost_class, expected_name",
        [
            (pybop.GravimetricEnergyDensity, "Gravimetric Energy Density"),
            (pybop.VolumetricEnergyDensity, "Volumetric Energy Density"),
            (pybop.GravimetricPowerDensity, "Gravimetric Power Density"),
            (pybop.VolumetricPowerDensity, "Volumetric Power Density"),
        ],
    )
    def test_design_costs(self, cost_class, expected_name, design_problem):
        # Construct Cost
        cost = cost_class(design_problem)
        assert cost.name == expected_name

        # Test type of returned value
        assert np.isscalar(cost([0.5]))
        assert cost([0.4]) >= 0  # Should be a viable design
        assert cost([0.8]) == -np.inf  # Should exceed active material + porosity < 1
        assert cost([1.4]) == -np.inf  # Definitely not viable
        assert cost([-0.1]) == -np.inf  # Should not be a viable design

        # Test infeasible locations
        cost.problem.model.allow_infeasible_solutions = False
        assert cost([1.1]) == -np.inf

        # Test exception for non-numeric inputs
        with pytest.raises(TypeError, match="Inputs must be a dictionary or numeric."):
            cost(["StringInputShouldNotWork"])

        # Compute after updating nominal capacity
        design_problem.update_capacity = True
        cost = cost_class(design_problem)
        cost([0.4])

    @pytest.fixture
    def noisy_problem(self, ground_truth, parameters, experiment):
        model = pybop.lithium_ion.SPM()
        model.parameter_set["Negative electrode active material volume fraction"] = (
            ground_truth
        )
        sol = model.predict(experiment=experiment)
        noisy_dataset = pybop.Dataset(
            {
                "Time [s]": sol["Time [s]"].data,
                "Current function [A]": sol["Current [A]"].data,
                "Voltage [V]": sol["Voltage [V]"].data
                + np.random.normal(0, 0.02, len(sol["Time [s]"].data)),
            }
        )
        return pybop.FittingProblem(model, parameters, noisy_dataset)

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
            cost1([0.6, sigma]) - 1 / weight * cost4([0.6, sigma]),
            atol=1e-5,
        )

    def test_weighted_design_cost(self, design_problem):
        cost1 = pybop.GravimetricEnergyDensity(design_problem)
        cost2 = pybop.VolumetricEnergyDensity(design_problem)

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
        cost3 = pybop.VolumetricEnergyDensity(copy(design_problem))
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

    def test_weighted_design_cost_with_update_capacity(self, design_problem):
        design_problem.update_capacity = True
        cost1 = pybop.GravimetricEnergyDensity(design_problem)
        cost2 = pybop.VolumetricEnergyDensity(design_problem)
        weighted_cost = pybop.WeightedCost(cost1, cost2, weights=[1, 1])

        assert weighted_cost.has_identical_problems is True
        assert weighted_cost.has_separable_problem is False
        assert weighted_cost.problem is design_problem
        assert weighted_cost([0.5]) >= 0
        np.testing.assert_allclose(
            weighted_cost([0.6]),
            cost1([0.6]) + cost2([0.6]),
            atol=1e-5,
        )

    def test_mixed_problem_classes(self, problem, design_problem):
        cost1 = pybop.SumSquaredError(problem)
        cost2 = pybop.GravimetricEnergyDensity(design_problem)
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
