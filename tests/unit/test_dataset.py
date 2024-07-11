import numpy as np
import pytest

import pybop


class TestDataset:
    """
    Class to test dataset construction.
    """

    @pytest.mark.unit
    def test_dataset(self):
        # Construct and simulate model
        model = pybop.lithium_ion.SPM()
        model.parameter_set = model.pybamm_model.default_parameter_values
        solution = model.predict(t_eval=np.linspace(0, 10, 100))

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
        assert dataset.names == pybop.Dataset(solution).names

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
