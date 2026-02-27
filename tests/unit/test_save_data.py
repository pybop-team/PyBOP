import numpy as np
import pytest

import pybop


class TestSaveData:
    """
    Class for unit testing saving data dictionaries.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def data_dict(self):
        return {
            "name": "some data",
            "float": 0.4,
            "list": [1.5, 5.6, 7.8],
            "numpy_array": np.linspace(0.5, 2.5, 10),
            "two_d_array": np.linspace([0.5, 1.0], [2.5, 4.0], 10),
        }

    @pytest.mark.parametrize("file_format", ["pickle", "json", "csv", "matlab"])
    def test_save_and_load_data(self, data_dict, file_format, tmp_path):
        test_stub = tmp_path / "test"

        # define appropriate file name
        if file_format == "matlab":
            filename = f"{test_stub}.mat"
        elif file_format == "json":
            filename = f"{test_stub}.json"
        elif file_format == "csv":
            # csv can only handle 1-d arrays of the same length
            data_dict = {
                "strings": ["word", "two words", "another string"],
                "list": [1.5, 5.6, 7.8],
                "numpy_array": np.linspace(0.5, 2.5, 3),
            }
            filename = f"{test_stub}.csv"
        else:
            filename = f"{test_stub}.pickle"

        # save data
        pybop.save_data_dict(data_dict, filename, to_format=file_format)

        # load data
        data_load = pybop.load_data_dict(filename, file_format=file_format)

        # compare original data with loaded data
        for key, data in data_dict.items():
            np.testing.assert_array_equal(
                np.asarray(data_load[key]).flatten(), np.asarray(data).flatten()
            )

        # check if string is the same as file for json and csv
        if file_format in ["csv", "json"]:
            data_str = pybop.save_data_dict(data_dict, to_format=file_format)

            # check string is the same as the file
            with open(filename) as f:
                # need to strip \r chars for windows
                assert data_str.replace("\r", "") == f.read()

        if file_format == "csv":
            # 1-d variables
            data_dict = {
                "string": "word",
                "number": 1.5,
            }
            # save data
            pybop.save_data_dict(data_dict, filename, to_format=file_format)

            # load data
            data_load = pybop.load_data_dict(
                filename, file_format=file_format, data_keys_0d=data_dict.keys()
            )

            # compare original data with loaded data
            for key, data in data_dict.items():
                assert data_load[key] == data

    def test_input_errors(self, data_dict, tmp_path):
        test_stub = tmp_path / "test"

        # raise error if filename not provided for pickle or matlab
        with pytest.raises(ValueError, match="pickle format must be written to a file"):
            pybop.save_data_dict(data_dict)

        with pytest.raises(ValueError, match="matlab format must be written to a file"):
            pybop.save_data_dict(data_dict, to_format="matlab")

        # raise error if format is unknown
        with pytest.raises(ValueError, match=r"format 'wrong_format' is not supported"):
            pybop.save_data_dict(data_dict, "{test_stub}.csv", to_format="wrong_format")

        with pytest.raises(ValueError, match=r"format 'wrong_format' is not supported"):
            pybop.load_data_dict("{test_stub}.csv", file_format="wrong_format")

        # To matlab with bad variable names fails
        data_dict = {
            "numpy array": np.linspace(0.5, 2.5, 3),
        }
        with pytest.raises(ValueError, match=r"Invalid character"):
            pybop.save_data_dict(data_dict, f"{test_stub}.mat", to_format="matlab")

        # raise error when saving multi-dimensional data to csv
        data_dict = {"two_d_array": np.linspace([0.5, 1.0], [2.5, 4.0], 10)}

        with pytest.raises(ValueError, match=r"only 0D"):
            pybop.save_data_dict(data_dict, f"{test_stub}.csv", to_format="csv")
