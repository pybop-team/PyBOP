import numpy as np
import pytest

import pybop
import pybamm


class TestDataset:
    """
    Class to test dataset construction.
    """

    pytestmark = pytest.mark.unit

    def test_dataset(self):
        # Construct and simulate model
        model = pybamm.lithium_ion.SPM()
        solution = pybamm.Simulation(model).solve(t_eval=[0, 3600])

        # Form dataset
        data_dictionary = {
            "Time [s]": solution["Time [s]"].data,
            "Current [A]": solution["Current [A]"].data,
            "Voltage [V]": solution["Voltage [V]"].data,
        }
        dataset = pybop.Dataset(data_dictionary)

        # Test repr
        print(dataset)

        # Test data structure
        assert dataset.data == data_dictionary
        assert np.all(dataset["Time [s]"] == solution["Time [s]"].data)

        # Test exception for non-dictionary inputs
        with pytest.raises(
            TypeError, match="The input to pybop.Dataset must be a dictionary."
        ):
            pybop.Dataset(["StringInputShouldNotWork"])
        with pytest.raises(
            TypeError, match="The input to pybop.Dataset must be a dictionary."
        ):
            pybop.Dataset(solution["Time [s]"].data)

        # Test conversion of pybamm solution into dictionary
        assert dataset.data == pybop.Dataset(solution).data

        # Test set and get item
        test_current = solution["Current [A]"].data + np.ones_like(
            solution["Current [A]"].data
        )
        dataset["Current [A]"] = test_current
        assert np.all(dataset["Current [A]"] == test_current)
        with pytest.raises(ValueError):
            dataset["Time"]

        # Test conversion of single signal to list
        assert dataset.check()

        # Test get subset
        dataset = dataset.get_subset(list(range(5)))
        assert len(dataset[dataset.domain]) == 5

        # Form frequency dataset
        data_dictionary = {
            "Frequency [Hz]": np.linspace(-10, 0, 10),
            "Current [A]": np.zeros(10),
            "Impedance": np.zeros(10),
        }
        frequency_dataset = pybop.Dataset(data_dictionary)

        with pytest.raises(ValueError, match="Frequencies cannot be negative."):
            frequency_dataset.check(domain="Frequency [Hz]", signal="Impedance")
