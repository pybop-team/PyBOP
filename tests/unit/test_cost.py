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
    def model(self):
        return pybamm.lithium_ion.SPM()

    @pytest.fixture
    def dataset(self):
        return pybop.Dataset(
            {
                "Time [s]": np.linspace(0, 360, 10),
                "Current function [A]": 1e-2 * np.ones(10),
                "Voltage [V]": np.ones(10),
            }
        )

    @pytest.fixture
    def one_parameter(self):
        return pybop.Parameter(
            "Negative electrode active material volume fraction",
            prior=pybop.Gaussian(0.5, 0.01),
            bounds=[0.375, 0.625],
        )

    @pytest.fixture
    def two_parameters(self):
        return [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.6, 0.02),
                bounds=[0.58, 0.62],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.05),
                bounds=[0.48, 0.52],
            ),
        ]

    @pytest.fixture
    def two_parameters_no_bounds(self):
        return [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                initial_value=0.6,
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                initial_value=0.5,
            ),
        ]

    @pytest.mark.parametrize(
        "pybamm_costs",
        [
            pybop.costs.pybamm.MeanSquaredError,
            pybop.costs.pybamm.RootMeanSquaredError,
            pybop.costs.pybamm.MeanAbsoluteError,
            pybop.costs.pybamm.SumSquaredError,
            pybop.costs.pybamm.Minkowski,
            pybop.costs.pybamm.SumOfPower,
            pybop.costs.pybamm.NegativeGaussianLogLikelihood,
        ],
    )
    def test_pybamm_costs(self, pybamm_costs, model, dataset, one_parameter):
        builder = pybop.Pybamm()
        builder.set_simulation(model)
        builder.set_dataset(dataset)
        builder.add_parameter(one_parameter)
        builder.add_cost(pybamm_costs("Voltage [V]", "Voltage [V]"))
        problem = builder.build()

        problem.set_params(np.array([0.55]))
        higher_cost = problem.run()
        problem.set_params(np.array([0.52]))
        lower_cost = problem.run()

        assert higher_cost > lower_cost

    def test_multi_cost_weighting(self, model, dataset, one_parameter):
        # ToDo: Add test for design cost as well
        def problem(weights):
            builder = pybop.Pybamm()
            builder.set_simulation(model)
            builder.set_dataset(dataset)
            builder.add_parameter(one_parameter)
            builder.add_cost(
                pybop.costs.pybamm.RootMeanSquaredError("Voltage [V]", "Voltage [V]"),
                weight=weights[0],
            )
            builder.add_cost(
                pybop.costs.pybamm.RootMeanSquaredError("Voltage [V]", "Voltage [V]"),
                weight=weights[1],
            )
            return builder.build()

        problem1 = problem([1, 1])
        problem2 = problem([1, 10])

        problem2.set_params(np.array([0.55]))
        val1 = problem1.run()
        problem2.set_params(np.array([0.55]))
        val2 = problem2.run()

        np.testing.assert_allclose(val2 / val1, 5.5)

    @pytest.mark.parametrize(
        "pybop_cost",
        [
            pybop.MeanSquaredError,
            pybop.RootMeanSquaredError,
            pybop.MeanAbsoluteError,
            pybop.SumSquaredError,
            pybop.Minkowski,
            pybop.SumOfPower,
            # pybop.NegativeGaussianLogLikelihood,
        ],
    )
    def test_error_measures(self, pybop_cost):
        res = np.asarray([i for i in range(10)])
        dy = np.ones((2, 10))

        # Test w/o sensitivities
        cost = pybop_cost(weighting=1)
        val = cost(res)
        assert val > 0

        # Test w/ sensitivities
        val, grad = cost(res, dy)
        assert val > 0
        assert grad > 0

        # # Test cost direction
        # higher_cost = cost([0.55])
        # lower_cost = cost([0.52])
        # assert higher_cost > lower_cost or (
        #     higher_cost == lower_cost and not np.isfinite(higher_cost)
        # )
        #
        # e, de = cost([0.5], calculate_grad=True)
        #
        # assert np.isscalar(e)
        # assert isinstance(de, np.ndarray)
        #
        # # Test exception for non-numeric inputs
        # with pytest.raises(TypeError, match="Inputs must be a dictionary or numeric."):
        #     cost(["StringInputShouldNotWork"], calculate_grad=True)
        #
        # with pytest.warns(UserWarning) as record:
        #     cost([1.1], calculate_grad=True)
        #
        # for i in range(len(record)):
        #     assert "Non-physical point encountered" in str(record[i].message)
        #
        # # Test infeasible locations
        # cost.problem.model.allow_infeasible_solutions = False
        # assert cost([1.1]) == np.inf
        # assert cost([1.1], calculate_grad=True) == (np.inf, cost._de)
        # assert cost([0.01]) == np.inf
        # assert cost([0.01], calculate_grad=True) == (np.inf, cost._de)

    def test_minkowski(self):
        # Incorrect order
        with pytest.raises(ValueError, match="The order of the Minkowski distance"):
            pybop.costs.pybamm.Minkowski("Voltage [V]", "Voltage [V]", p=-1)
        with pytest.raises(
            ValueError,
            match="For p = infinity, an implementation of the Chebyshev distance is required.",
        ):
            pybop.costs.pybamm.Minkowski("Voltage [V]", "Voltage [V]", p=np.inf)

    def test_sumofpower(self, problem):
        # Incorrect order
        with pytest.raises(
            ValueError, match="The order of 'p' must be greater than 0."
        ):
            pybop.costs.pybamm.Minkowski("Voltage [V]", "Voltage [V]", p=-1)

        with pytest.raises(ValueError, match="p = np.inf is not yet supported."):
            pybop.costs.pybamm.Minkowski("Voltage [V]", "Voltage [V]", p=np.inf)

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
