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
    def model(self):
        return pybop.lithium_ion.SPM()

    @pytest.fixture
    def ground_truth(self):
        return 0.52

    @pytest.fixture
    def parameters(self, ground_truth):
        return pybop.Parameter(
            "Negative electrode active material volume fraction",
            prior=pybop.Gaussian(0.5, 0.01),
            bounds=[0.375, 0.625],
            initial_value=ground_truth,
        )

    @pytest.fixture
    def experiment(self):
        return pybop.Experiment(
            [
                ("Discharge at 1C for 10 minutes (20 second period)"),
            ]
        )

    @pytest.fixture
    def dataset(self, model, experiment, ground_truth):
        model.parameter_set = model.pybamm_model.default_parameter_values
        model.parameter_set.update(
            {
                "Negative electrode active material volume fraction": ground_truth,
            }
        )
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
        problem = pybop.FittingProblem(
            model, parameters, dataset, signal=signal, init_soc=1.0
        )
        problem.dataset = dataset  # add this to pass the pybop dataset to cost
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
            state = problem._model.reinit(inputs)
            n = len(state)
            sigma_diag = [0.0] * n
            sigma_diag[0] = 1e-4
            sigma_diag[1] = 1e-4
            process_diag = [0.0] * n
            process_diag[0] = 1e-4
            process_diag[1] = 1e-4
            sigma0 = np.diag(sigma_diag)
            process = np.diag(process_diag)
            dataset = problem.dataset
            return cls(
                pybop.UnscentedKalmanFilterObserver(
                    problem.parameters,
                    problem._model,
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
        with pytest.raises(NotImplementedError):
            base_cost.evaluateS1([0.5])

    @pytest.mark.unit
    def test_error_in_cost_calculation(self, problem):
        class RaiseErrorCost(pybop.BaseCost):
            def _evaluate(self, inputs, grad=None):
                raise ValueError("Error test.")

            def _evaluateS1(self, inputs):
                raise ValueError("Error test.")

        cost = RaiseErrorCost(problem)
        with pytest.raises(ValueError, match="Error in cost calculation: Error test."):
            cost([0.5])
        with pytest.raises(ValueError, match="Error in cost calculation: Error test."):
            cost.evaluateS1([0.5])

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
        problem_non_finite = pybop.FittingProblem(
            problem.model, parameter, problem.dataset, signal=problem.signal
        )
        likelihood = pybop.MAP(
            problem_non_finite, pybop.GaussianLogLikelihoodKnownSigma, sigma0=0.01
        )
        assert not np.isfinite(likelihood([0.7]))
        assert not np.isfinite(likelihood.evaluateS1([0.7])[0])

    @pytest.mark.unit
    def test_costs(self, cost):
        if isinstance(cost, pybop.BaseLikelihood):
            higher_cost = cost([0.52])
            lower_cost = cost([0.55])
        else:
            higher_cost = cost([0.55])
            lower_cost = cost([0.52])
        assert higher_cost > lower_cost or (
            higher_cost == lower_cost and higher_cost == np.inf
        )

        # Test type of returned value
        assert np.isscalar(cost([0.5]))

        if isinstance(cost, pybop.ObserverCost):
            with pytest.raises(NotImplementedError):
                cost.evaluateS1([0.5])

        # Test UserWarnings
        if isinstance(cost, (pybop.SumSquaredError, pybop.RootMeanSquaredError)):
            assert cost([0.5]) >= 0
            with pytest.warns(UserWarning) as record:
                cost([1.1])

            # Test option setting
            cost.set_fail_gradient(1)

        if not isinstance(cost, (pybop.ObserverCost, pybop.MAP)):
            e, de = cost.evaluateS1([0.5])

            assert np.isscalar(e)
            assert isinstance(de, np.ndarray)

            # Test exception for non-numeric inputs
            with pytest.raises(
                TypeError, match="Inputs must be a dictionary or numeric."
            ):
                cost.evaluateS1(["StringInputShouldNotWork"])

            with pytest.warns(UserWarning) as record:
                cost.evaluateS1([1.1])

            for i in range(len(record)):
                assert "Non-physical point encountered" in str(record[i].message)

            # Test infeasible locations
            cost.problem._model.allow_infeasible_solutions = False
            assert cost([1.1]) == np.inf
            assert cost.evaluateS1([1.1]) == (np.inf, cost._de)
            assert cost([0.01]) == np.inf
            assert cost.evaluateS1([0.01]) == (np.inf, cost._de)

        # Test exception for non-numeric inputs
        with pytest.raises(TypeError, match="Inputs must be a dictionary or numeric."):
            cost(["StringInputShouldNotWork"])

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
    def test_SumofPower(self, problem):
        # Incorrect order
        with pytest.raises(
            ValueError, match="The order of 'p' must be greater than 0."
        ):
            pybop.SumofPower(problem, p=-1)

        with pytest.raises(ValueError, match="p = np.inf is not yet supported."):
            pybop.SumofPower(problem, p=np.inf)

    @pytest.mark.parametrize(
        "cost_class",
        [
            pybop.DesignCost,
            pybop.GravimetricEnergyDensity,
            pybop.VolumetricEnergyDensity,
        ],
    )
    @pytest.mark.unit
    def test_design_costs(
        self,
        cost_class,
        model,
        parameters,
        experiment,
        signal,
    ):
        # Construct Problem
        problem = pybop.DesignProblem(
            model, parameters, experiment, signal=signal, init_soc=0.5
        )

        # Construct Cost
        cost = cost_class(problem)

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
            cost.problem._model.allow_infeasible_solutions = False
            assert cost([1.1]) == -np.inf

            # Test exception for non-numeric inputs
            with pytest.raises(
                TypeError, match="Inputs must be a dictionary or numeric."
            ):
                cost(["StringInputShouldNotWork"])

            # Compute after updating nominal capacity
            cost = cost_class(problem, update_capacity=True)
            cost([0.4])
