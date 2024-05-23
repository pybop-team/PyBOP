import numpy as np
import pytest

import pybop


class TestCosts:
    """
    Class for tests cost functions
    """

    @pytest.fixture
    def model(self):
        return pybop.lithium_ion.SPM()

    @pytest.fixture
    def parameters(self):
        return [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.5, 0.01),
                bounds=[0.375, 0.625],
            ),
        ]

    @pytest.fixture
    def experiment(self):
        return pybop.Experiment(
            [
                ("Discharge at 1C for 10 minutes (20 second period)"),
            ]
        )

    @pytest.fixture
    def x0(self):
        return np.array([0.52])

    @pytest.fixture
    def dataset(self, model, experiment, x0):
        model.parameter_set = model.pybamm_model.default_parameter_values
        model.parameter_set.update(
            {
                "Negative electrode active material volume fraction": x0[0],
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
    def problem(self, model, parameters, dataset, signal, x0, request):
        cut_off = request.param
        model.parameter_set.update({"Lower voltage cut-off [V]": cut_off})
        problem = pybop.FittingProblem(
            model, parameters, dataset, signal=signal, x0=x0, init_soc=1.0
        )
        problem.dataset = dataset  # add this to pass the pybop dataset to cost
        return problem

    @pytest.fixture(
        params=[
            pybop.RootMeanSquaredError,
            pybop.SumSquaredError,
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
        elif cls in [pybop.ObserverCost]:
            inputs = {p.name: problem.x0[i] for i, p in enumerate(problem.parameters)}
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
    def test_design_base(self, problem):
        design_cost = pybop.DesignCost(problem)
        with pytest.raises(NotImplementedError):
            design_cost([0.5])

    @pytest.mark.unit
    def test_MAP(self, problem):
        # Incorrect likelihood
        with pytest.raises(ValueError):
            pybop.MAP(problem, pybop.SumSquaredError)

        # Incorrect construction of likelihood
        with pytest.raises(ValueError):
            pybop.MAP(problem, pybop.GaussianLogLikelihoodKnownSigma, sigma="string")

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

        if isinstance(cost, pybop.SumSquaredError):
            e, de = cost.evaluateS1([0.5])

            assert np.isscalar(e)
            assert type(de) == np.ndarray

            # Test exception for non-numeric inputs
            with pytest.raises(ValueError):
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
        with pytest.raises(ValueError):
            cost(["StringInputShouldNotWork"])

        # Test treatment of simulations that terminated early
        # by variation of the cut-off voltage.

    @pytest.mark.parametrize(
        "cost_class",
        [pybop.GravimetricEnergyDensity, pybop.VolumetricEnergyDensity],
    )
    @pytest.mark.unit
    def test_energy_density_costs(
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

        # Test type of returned value
        assert np.isscalar(cost([0.5]))
        assert cost([0.4]) >= 0  # Should be a viable design
        assert cost([0.8]) == -np.inf  # Should exceed active material + porosity < 1
        assert cost([1.4]) == -np.inf  # Definitely not viable
        assert cost([-0.1]) == -np.inf  # Should not be a viable design

        # Test infeasible locations
        cost.problem._model.allow_infeasible_solutions = False
        assert cost([1.1]) == -np.inf

        # Test exception for non-numeric inputs
        with pytest.raises(ValueError):
            cost(["StringInputShouldNotWork"])

        # Compute after updating nominal capacity
        cost = cost_class(problem, update_capacity=True)
        cost([0.4])
