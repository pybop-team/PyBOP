from copy import copy

import numpy as np
import pytest

import pybop


class TestCosts:
    """
    Class for tests cost functions
    """

    # Define an invalid likelihood class for MAP tests
    class InvalidLikelihood:
        def __init__(self, problem, sigma0):
            pass

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
        return pybop.Parameter(
            "Negative electrode active material volume fraction",
            prior=pybop.Gaussian(0.5, 0.01),
            bounds=[0.375, 0.625],
        )

    @pytest.fixture
    def experiment(self):
        return pybop.Experiment(
            [
                ("Discharge at 1C for 10 minutes (20 second period)"),
            ]
        )

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

    @pytest.fixture
    def signal(self):
        return "Voltage [V]"

    @pytest.fixture(params=[2.5, 3.777])
    def problem(self, model, parameters, dataset, signal, request):
        cut_off = request.param
        model.parameter_set.update({"Lower voltage cut-off [V]": cut_off})
        problem = pybop.FittingProblem(model, parameters, dataset, signal=signal)
        return problem

    @pytest.fixture(
        params=[
            pybop.RootMeanSquaredError,
            pybop.SumSquaredError,
            pybop.Minkowski,
            pybop.SumofPower,
            pybop.ObserverCost,
            pybop.MAP,
        ]
    )
    def cost(self, problem, request):
        cls = request.param
        if cls in [pybop.SumSquaredError, pybop.RootMeanSquaredError]:
            return cls(problem)
        elif cls in [pybop.MAP]:
            return cls(problem, pybop.GaussianLogLikelihoodKnownSigma)
        elif cls in [pybop.Minkowski, pybop.SumofPower]:
            return cls(problem, p=2)
        elif cls in [pybop.ObserverCost]:
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
                    signal=problem.signal,
                ),
            )

    @pytest.mark.unit
    def test_base(self, problem):
        base_cost = pybop.BaseCost(problem)
        assert base_cost.problem == problem
        with pytest.raises(NotImplementedError):
            base_cost([0.5])

    @pytest.mark.unit
    def test_error_in_cost_calculation(self, problem):
        class RaiseErrorCost(pybop.BaseCost):
            def compute(self, y, dy=None, calculate_grad: bool = False):
                raise ValueError("Error test.")

        cost = RaiseErrorCost(problem)
        with pytest.raises(ValueError, match="Error in cost calculation: Error test."):
            cost([0.5])
        with pytest.raises(ValueError, match="Error in cost calculation: Error test."):
            cost([0.5], calculate_grad=True)

    @pytest.mark.unit
    def test_MAP(self, problem):
        # Incorrect likelihood
        with pytest.raises(
            ValueError,
            match="An error occurred when constructing the Likelihood class:",
        ):
            pybop.MAP(problem, pybop.SumSquaredError)

        # Incorrect construction of likelihood
        with pytest.raises(
            ValueError,
            match="An error occurred when constructing the Likelihood class: could not convert string to float: 'string'",
        ):
            pybop.MAP(problem, pybop.GaussianLogLikelihoodKnownSigma, sigma0="string")

        # Incorrect likelihood
        with pytest.raises(ValueError, match="must be a subclass of BaseLikelihood"):
            pybop.MAP(problem, self.InvalidLikelihood, sigma0=0.1)

        # Non finite prior
        parameter = pybop.Parameter(
            "Negative electrode active material volume fraction",
            prior=pybop.Uniform(0.55, 0.6),
        )
        dataset = pybop.Dataset(data_dictionary=problem.dataset)
        problem_non_finite = pybop.FittingProblem(
            problem.model, parameter, dataset, signal=problem.signal
        )
        likelihood = pybop.MAP(
            problem_non_finite, pybop.GaussianLogLikelihoodKnownSigma, sigma0=0.01
        )
        assert not np.isfinite(likelihood([0.7]))
        assert not np.isfinite(likelihood([0.7], calculate_grad=True)[0])

    @pytest.mark.unit
    def test_costs(self, cost):
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

        # Test UserWarnings
        if isinstance(cost, (pybop.SumSquaredError, pybop.RootMeanSquaredError)):
            assert cost([0.5]) >= 0
            with pytest.warns(UserWarning) as record:
                cost([1.1])

            # Test option setting
            cost.set_fail_gradient(10)
            assert cost._de == 10

        if not isinstance(cost, (pybop.ObserverCost, pybop.MAP)):
            e, de = cost([0.5], calculate_grad=True)

            assert np.isscalar(e)
            assert isinstance(de, np.ndarray)

            # Test exception for non-numeric inputs
            with pytest.raises(
                TypeError, match="Inputs must be a dictionary or numeric."
            ):
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

        # Test exception for non-numeric inputs
        with pytest.raises(TypeError, match="Inputs must be a dictionary or numeric."):
            cost(["StringInputShouldNotWork"])

        # Test ValueError for none dy w/ calculate_grad == True
        if not isinstance(cost, pybop.ObserverCost):
            with pytest.raises(
                ValueError,
                match="Forward model sensitivities need to be provided alongside `calculate_grad=True` for `cost.compute`.",
            ):
                cost.compute([1.1], dy=None, calculate_grad=True)

    @pytest.mark.unit
    def test_minkowski(self, problem):
        # Incorrect order
        with pytest.raises(ValueError, match="The order of the Minkowski distance"):
            pybop.Minkowski(problem, p=-1)
        with pytest.raises(
            ValueError,
            match="For p = infinity, an implementation of the Chebyshev distance is required.",
        ):
            pybop.Minkowski(problem, p=np.inf)

    @pytest.mark.unit
    def test_sumofpower(self, problem):
        # Incorrect order
        with pytest.raises(
            ValueError, match="The order of 'p' must be greater than 0."
        ):
            pybop.SumofPower(problem, p=-1)

        with pytest.raises(ValueError, match="p = np.inf is not yet supported."):
            pybop.SumofPower(problem, p=np.inf)

    @pytest.fixture
    def design_problem(self, model, parameters, experiment, signal):
        return pybop.DesignProblem(
            model,
            parameters,
            experiment,
            signal=signal,
            initial_state={"Initial SoC": 0.5},
        )

    @pytest.mark.parametrize(
        "cost_class",
        [
            pybop.DesignCost,
            pybop.GravimetricEnergyDensity,
            pybop.VolumetricEnergyDensity,
        ],
    )
    @pytest.mark.unit
    def test_design_costs(self, cost_class, design_problem):
        # Construct Cost
        cost = cost_class(design_problem)

        if cost_class in [pybop.DesignCost]:
            with pytest.raises(NotImplementedError):
                cost([0.5])
        else:
            # Test type of returned value
            assert np.isscalar(cost([0.5]))
            assert cost([0.4]) >= 0  # Should be a viable design
            assert (
                cost([0.8]) == -np.inf
            )  # Should exceed active material + porosity < 1
            assert cost([1.4]) == -np.inf  # Definitely not viable
            assert cost([-0.1]) == -np.inf  # Should not be a viable design

            # Test infeasible locations
            cost.problem.model.allow_infeasible_solutions = False
            assert cost([1.1]) == -np.inf

            # Test exception for non-numeric inputs
            with pytest.raises(
                TypeError, match="Inputs must be a dictionary or numeric."
            ):
                cost(["StringInputShouldNotWork"])

            # Compute after updating nominal capacity
            cost = cost_class(design_problem, update_capacity=True)
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

    @pytest.mark.unit
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
            weighted_cost = pybop.WeightedCost(cost1.problem)
        with pytest.raises(
            ValueError,
            match="Weights must be numeric values.",
        ):
            weighted_cost = pybop.WeightedCost(cost1, cost2, weights="Invalid string")
        with pytest.raises(
            ValueError,
            match="Number of weights must match number of costs.",
        ):
            weighted_cost = pybop.WeightedCost(cost1, cost2, weights=[1])

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

        # Test MAP explicitly
        cost4 = pybop.MAP(problem, pybop.GaussianLogLikelihood)
        weighted_cost_4 = pybop.WeightedCost(cost1, cost4, weights=[1, -1 / weight])
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

    @pytest.mark.unit
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

        # Ensure attributes are set correctly and not modified via side-effects
        assert weighted_cost.update_capacity is False
        assert weighted_cost.costs[0].update_capacity is False
        assert weighted_cost.costs[1].update_capacity is False

        assert weighted_cost([0.5]) >= 0
        np.testing.assert_allclose(
            weighted_cost([0.6]),
            cost1([0.6]) + cost2([0.6]),
            atol=1e-5,
        )

    @pytest.mark.unit
    def test_weighted_design_cost_with_update_capacity(self, design_problem):
        cost1 = pybop.GravimetricEnergyDensity(design_problem, update_capacity=True)
        cost2 = pybop.VolumetricEnergyDensity(design_problem)
        weighted_cost = pybop.WeightedCost(cost1, cost2, weights=[1, 1])

        # Ensure attributes are set correctly and not modified via side-effects
        assert weighted_cost.update_capacity is True
        assert weighted_cost.costs[0].update_capacity is True
        assert weighted_cost.costs[1].update_capacity is False

        assert weighted_cost.has_identical_problems is True
        assert weighted_cost.has_separable_problem is False
        assert weighted_cost.problem is design_problem
        assert weighted_cost([0.5]) >= 0
        np.testing.assert_allclose(
            weighted_cost([0.6]),
            cost1([0.6]) + cost2([0.6]),
            atol=1e-5,
        )

    @pytest.mark.unit
    def test_mixed_problem_classes(self, problem, design_problem):
        cost1 = pybop.SumSquaredError(problem)
        cost2 = pybop.GravimetricEnergyDensity(design_problem)
        with pytest.raises(
            TypeError,
            match="All problems must be of the same class type.",
        ):
            pybop.WeightedCost(cost1, cost2)
