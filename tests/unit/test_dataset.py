import pytest
import pybop
import numpy as np


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
            "Terminal voltage [V]": solution["Terminal voltage [V]"].data,
        }
        dataset = pybop.Dataset(data_dictionary)

        # Test repr
        print(dataset)

        # Test data structure
        assert dataset.data == data_dictionary
        assert np.all(dataset["Time [s]"] == solution["Time [s]"].data)

        # Test exception for non-dictionary inputs
        with pytest.raises(ValueError):
            pybop.Dataset(["StringInputShouldNotWork"])
        with pytest.raises(ValueError):
            pybop.Dataset(solution["Time [s]"].data)

        # Test conversion of pybamm solution into dictionary
        assert dataset.data == pybop.Dataset(solution).data
        assert dataset.names == pybop.Dataset(solution).names
