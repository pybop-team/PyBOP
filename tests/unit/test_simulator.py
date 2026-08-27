from copy import copy

import numpy as np
import pybamm
import pytest

import pybop
from pybop.simulators.base_simulator import BaseSimulator


class TestSimulator:
    """
    A class to test the BaseSimulator class.
    """

    pytestmark = pytest.mark.unit

    def test_parameter_errors_constructor(self):
        params = {
            "Negative particle radius [m]": pybop.Parameter(
                pybop.Gaussian(2e-05, 1e-6, truncated_at=[1e-6, 5e-5])
            ),
            "Positive particle radius [m]": pybop.Parameter(
                pybop.Gaussian(0.5e-05, 1e-6, truncated_at=[1e-6, 5e-5])
            ),
        }

        simulator = BaseSimulator(copy(params))
        for key in params.keys():
            assert simulator.parameters[key] is params[key]
        # Check that pybop.Parameter objects are removed from the dictionary
        simulator = BaseSimulator(params)
        for key in params.keys():
            assert params[key] == "[input]"

        params = {
            "Negative particle radius [m]": (
                pybop.Gaussian(2e-05, 0.1e-5, truncated_at=[1e-6, 5e-5])
            ),
            "Positive particle radius [m]": (
                pybop.Gaussian(0.5e-05, 0.1e-5, truncated_at=[1e-6, 5e-5])
            ),
        }

        simulator = BaseSimulator(copy(params))
        assert len(simulator.parameters) == 0

        params = [
            pybop.Parameter(pybop.Gaussian(2e-05, 1e-6, truncated_at=[1e-6, 5e-5])),
            pybop.Parameter(pybop.Gaussian(2e-05, 1e-6, truncated_at=[1e-6, 5e-5])),
        ]

        with pytest.raises(
            TypeError,
            match="Parameters must be a dictionary of pybop.Parameter objects or a pybop.Parameters object.",
        ):
            BaseSimulator(params)


class TestPybammSimulator:
    """
    A class to test the pybamm.Simulator class.
    """

    pytestmark = pytest.mark.unit

    def test_set_output_variables(self):
        model = pybamm.lithium_ion.SPM()
        parameter_values = model.default_parameter_values
        experiment = pybamm.Experiment(
            ["Discharge at 1C for 5 minutes (10 second period)"]
        )

        simulator = pybop.pybamm.Simulator(
            model, parameter_values=parameter_values, protocol=experiment
        )

        with pytest.raises(
            ValueError, match="Not a variable is not a variable in the model."
        ):
            simulator.set_output_variables(["Not a variable"])


class TestCoupledEISSimulator:
    """
    A class to test the pybamm.EISSimulator class when coupled to a protocol.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def model(self):
        return pybamm.lithium_ion.SPM(options={"surface form": "differential"})

    @pytest.fixture
    def parameter_values(self):
        parameter_values = pybamm.ParameterValues("Chen2020")
        parameter_values.set_initial_state(0.9)
        return parameter_values

    @pytest.fixture
    def frequencies(self):
        return np.logspace(-1, 3, 4)

    @pytest.fixture
    def impedance_variables(self, frequencies):
        return pybop.get_impedance_variables(frequencies)

    @pytest.fixture
    def acquisitions(self):
        """The rows at which a spectrum is acquired: one at the start, one at rest."""
        return [0, 40]

    @pytest.fixture
    def dataset(self, impedance_variables, acquisitions):
        time = np.arange(0, 601, 10.0)
        data = {
            "Time [s]": time,
            "Current [A]": np.zeros_like(time),
            "Voltage [V]": np.zeros_like(time),
        }
        for name in impedance_variables:
            variable = np.zeros(len(time))
            variable[acquisitions] = 1.0
            data[name] = variable

        return pybop.Dataset(data, domain="Time [s]")

    def test_output_shape_and_zero_padding(
        self,
        model,
        parameter_values,
        dataset,
        frequencies,
        impedance_variables,
        acquisitions,
    ):
        simulator = pybop.pybamm.EISSimulator(
            model,
            parameter_values=parameter_values,
            protocol=dataset,
            f_eval=frequencies,
        )
        solution = simulator.solve()

        n_time = len(dataset["Time [s]"])
        assert len(solution["Voltage [V]"].data) == n_time
        for name in impedance_variables:
            data = solution[name].data
            assert len(data) == n_time
            # Non-zero only at the times of acquisition
            np.testing.assert_array_equal(np.flatnonzero(data), acquisitions)

    def test_matches_stationary_at_initial_state(
        self, model, parameter_values, dataset, frequencies, impedance_variables
    ):
        """The spectrum at t=0 must match a stationary simulation of the same state."""
        coupled = pybop.pybamm.EISSimulator(
            model,
            parameter_values=parameter_values,
            protocol=dataset,
            f_eval=frequencies,
        ).solve()
        stationary = pybop.pybamm.EISSimulator(
            model, parameter_values=parameter_values, f_eval=frequencies
        ).solve()

        impedance = np.asarray(
            [
                coupled[impedance_variables[2 * j]].data[0]
                + 1j * coupled[impedance_variables[2 * j + 1]].data[0]
                for j in range(len(frequencies))
            ]
        )
        np.testing.assert_allclose(impedance, stationary["Impedance"].data, rtol=1e-10)

    @pytest.mark.parametrize("coupled", [False, True])
    def test_input_parameters_are_not_swapped(
        self,
        model,
        parameter_values,
        dataset,
        frequencies,
        impedance_variables,
        coupled,
    ):
        """
        Passing values as inputs must give the same spectrum as fixing them in the
        parameter values, whatever the order of the inputs dictionary. The compiled
        Jacobian takes a single stacked vector, so an order which disagrees with the
        one used to compile it silently evaluates the model at the wrong point.
        """
        # Deliberately not in alphabetical order
        inputs = {
            "Positive particle diffusivity [m2.s-1]": 4e-15,
            "Negative particle diffusivity [m2.s-1]": 3.0e-14,
        }

        def solve(values, **kwargs):
            protocol = dataset if coupled else None
            simulator = pybop.pybamm.EISSimulator(
                model,
                parameter_values=values,
                protocol=protocol,
                f_eval=frequencies,
            )
            return simulator.solve(**kwargs)

        fixed = copy(parameter_values)
        fixed.update(inputs)
        expected = solve(fixed)

        fitted = copy(parameter_values)
        fitted.update(
            {
                name: pybop.Parameter(pybop.Uniform(value / 2, value * 2))
                for name, value in inputs.items()
            }
        )
        solution = solve(fitted, inputs=inputs)

        if coupled:
            for name in impedance_variables:
                np.testing.assert_allclose(
                    solution[name].data, expected[name].data, rtol=1e-10
                )
        else:
            np.testing.assert_allclose(
                solution["Impedance"].data, expected["Impedance"].data, rtol=1e-10
            )

    def test_geometric_parameter_rebuilds(
        self, model, parameter_values, dataset, frequencies, impedance_variables
    ):
        """
        A geometric parameter forces a rebuild on every evaluation, which discards the
        simulation, so the constant matrices must be set up after the time-domain solve.
        """
        inputs = {"Negative electrode thickness [m]": 8.0e-05}

        fixed = copy(parameter_values)
        fixed.update(inputs)
        expected = pybop.pybamm.EISSimulator(
            model, parameter_values=fixed, protocol=dataset, f_eval=frequencies
        ).solve()

        fitted = copy(parameter_values)
        fitted.update(
            {
                name: pybop.Parameter(pybop.Uniform(value / 2, value * 2))
                for name, value in inputs.items()
            }
        )
        simulator = pybop.pybamm.EISSimulator(
            model, parameter_values=fitted, protocol=dataset, f_eval=frequencies
        )
        assert simulator._simulator.requires_model_rebuild  # noqa: SLF001
        solution = simulator.solve(inputs)

        for name in impedance_variables:
            np.testing.assert_allclose(
                solution[name].data, expected[name].data, rtol=1e-10
            )

    def test_builds_once(self, model, parameter_values, dataset, frequencies):
        """The model and the constant matrices are set up once, not per evaluation."""
        parameter_values["Negative electrode active material volume fraction"] = (
            pybop.Parameter(pybop.Uniform(0.4, 0.75))
        )
        simulator = pybop.pybamm.EISSimulator(
            model,
            parameter_values=parameter_values,
            protocol=dataset,
            f_eval=frequencies,
        )

        calls = {"create": 0, "set_up": 0}
        original_create = simulator._simulator.create_simulation
        original_set_up = simulator._set_up_matrices

        def counted_create(*args, **kwargs):
            calls["create"] += 1
            return original_create(*args, **kwargs)

        def counted_set_up(*args, **kwargs):
            calls["set_up"] += 1
            return original_set_up(*args, **kwargs)

        simulator._simulator.create_simulation = counted_create
        simulator._set_up_matrices = counted_set_up

        inputs = {"Negative electrode active material volume fraction": 0.6}
        for _ in range(3):
            simulator.solve(inputs)

        assert calls["create"] == 0  # built during construction
        assert calls["set_up"] == 1

    def test_model_is_not_modified(self, model, parameter_values, dataset, frequencies):
        """Setting up for EIS must copy the model, not modify the caller's."""
        n_algebraic = len(model.algebraic)
        pybop.pybamm.EISSimulator(
            model,
            parameter_values=parameter_values,
            protocol=dataset,
            f_eval=frequencies,
        )
        assert len(model.algebraic) == n_algebraic

    def test_model_without_surface_form(
        self, parameter_values, dataset, frequencies, impedance_variables
    ):
        """Without a surface form there is no double layer, but the diffusion tail is
        still a valid thing to simulate, so the model must warn rather than be rejected."""
        with pytest.warns(UserWarning, match="charge-transfer arc"):
            simulator = pybop.pybamm.EISSimulator(
                pybamm.lithium_ion.SPM(),
                parameter_values=parameter_values,
                protocol=dataset,
                f_eval=frequencies,
            )
        solution = simulator.solve()

        for name in impedance_variables:
            assert np.all(np.isfinite(solution[name].data))

    def test_impedance_variables_round_trip(self, frequencies, impedance_variables):
        # Variables may reach the parser in any order, e.g. via a set
        parsed, real, imaginary = pybop.parse_impedance_variables(
            ["Voltage [V]", *reversed(impedance_variables)]
        )
        # The names carry six significant figures, which bounds the round trip
        np.testing.assert_allclose(parsed, frequencies, rtol=1e-5)
        assert real == impedance_variables[::2]
        assert imaginary == impedance_variables[1::2]

        # No impedance variables present
        parsed, real, imaginary = pybop.parse_impedance_variables(
            ["Voltage [V]", "Current [A]"]
        )
        assert len(parsed) == 0 and real == [] and imaginary == []

    def test_dataset_errors(
        self, model, parameter_values, dataset, frequencies, impedance_variables
    ):
        with pytest.raises(ValueError, match="missing impedance variables"):
            pybop.pybamm.EISSimulator(
                model,
                parameter_values=parameter_values,
                protocol=dataset,
                f_eval=np.append(frequencies, 1e4),
            )

        for name in impedance_variables:
            dataset[name] = np.zeros(len(dataset["Time [s]"]))
        with pytest.raises(ValueError, match="zero everywhere"):
            pybop.pybamm.EISSimulator(
                model,
                parameter_values=parameter_values,
                protocol=dataset,
                f_eval=frequencies,
            )
