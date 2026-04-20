import json
import pickle
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat


class NumpyEncoder(json.JSONEncoder):
    """
    Numpy serialiser helper class that converts numpy arrays to a list.
    Numpy arrays cannot be directly converted to JSON, so the arrays are
    converted to python list objects before encoding.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # won't be called since we only need to convert numpy arrays
        return json.JSONEncoder.default(self, obj)  # pragma: no cover


def add_spaces(string):
    """
    Return the class name as a string with spaces before each new capitalised word.
    """
    re_outer = re.compile(r"([^A-Z ])([A-Z])")
    re_inner = re.compile(r"(?<!^)([A-Z])([^A-Z])")
    return re_outer.sub(r"\1 \2", re_inner.sub(r" \1\2", string))


def add_noise(y, sigma):
    """
    Add random Gaussian noise to an array.
    """
    if np.iscomplexobj(y):
        # Add noise in the imaginary component as well as the real
        y += 1j * np.random.normal(loc=0.0, scale=sigma, size=np.shape(y))
    return y + np.random.normal(loc=0.0, scale=sigma, size=np.shape(y))


def is_numeric(x):
    """
    Check if a variable is numeric.
    """
    return isinstance(x, int | float | np.number)


def save_data_dict(
    data_dict: dict,
    filename: str | None = None,
    to_format: str = "pickle",
) -> str | None:
    """
    Save data from given data dictionary

    Based on pybamm.Solution.save_data

    Parameters
    ----------
    filename : str, optional
        The name of the file to save data to. If None, then a str is returned
    to_format : str, optional
        The format to save to. Options are:

        - 'pickle' (default): creates a pickle file with the data dictionary
        - 'matlab': creates a .mat file, for loading in matlab
        - 'csv': creates a csv file (0D variables only)
        - 'json': creates a json file

    Returns
    -------
    data : str, optional
        str if 'json' is chosen and filename is None, otherwise None
    """

    if to_format == "pickle":
        if filename is None:
            raise ValueError("pickle format must be written to a file")
        with open(filename, "wb") as f:
            pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)
    elif to_format == "matlab":
        if filename is None:
            raise ValueError("matlab format must be written to a file")
        # Check all the variable names only contain a-z, A-Z or _ or numbers
        for name in data_dict.keys():
            # Check the string only contains the following ASCII:
            # a-z (97-122)
            # A-Z (65-90)
            # _ (95)
            # 0-9 (48-57) but not in the first position
            for i, s in enumerate(name):
                if not (
                    97 <= ord(s) <= 122
                    or 65 <= ord(s) <= 90
                    or ord(s) == 95
                    or (i > 0 and 48 <= ord(s) <= 57)
                ):
                    raise ValueError(
                        f"Invalid character '{s}' found in '{name}'. "
                        "MATLAB variable names must only contain a-z, A-Z, _, "
                        "or 0-9 (except the first position). "
                    )
        savemat(filename, data_dict)
    elif to_format == "csv":
        # use copy to avoid modifying input
        data_dict_copy = data_dict.copy()
        for name, var in data_dict.items():
            var = np.asarray(var)
            if var.ndim == 0:
                data_dict_copy[name] = [var]
            elif var.ndim >= 2:
                raise ValueError(
                    f"only 0D variables can be saved to csv, but '{name}' is {var.ndim - 1}D"
                )
        df = pd.DataFrame(data_dict_copy)
        return df.to_csv(filename, index=False)
    elif to_format == "json":
        if filename is None:
            return json.dumps(data_dict, cls=NumpyEncoder)
        else:
            with open(filename, "w") as outfile:
                json.dump(data_dict, outfile, cls=NumpyEncoder)
    else:
        raise ValueError(f"format '{to_format}' is not supported")


def load_data_dict(
    filename: str,
    file_format: str = "pickle",
    data_keys_0d: list[str] | None = None,
    data_keys_1d: list[str] | None = None,
) -> dict:
    """
    Load data as dictionary from a given file. Restores data saved with
    save_data_dict.

    Parameters
    ----------
    filename : str
        The name of the file containing the data.
    file_format : str, optional
        The format the data was save to. Options are:
        - 'pickle' (default)
        - 'matlab'
        - 'csv'
        - 'json'
    data_keys_0d: list[str], optional
        A list of keys for which the data is a scalar/0-dimensional.
        This is only needed for file_format='matlab' or file_format = 'csv'.
        scipy.io.savemat turns any data into a multi-dimensional array with at least 2 dimensions.
        If provided, data dimensions will be consistent with the original data.
    data_keys_1d: list[str], optional
        A list of keys for which the data is a 1-dimensional list or array.
        This is only needed for file_format='matlab'. scipy.io.savemat turns any
        data into a multi-dimensional array with at least 2 dimensions.
        If provided, data dimensions will be consistent with the original data.

    Returns
    -------
    data_dict :
        python dictionary containing the data in the file.
    """

    # Load data using appropriate method according to the file_format
    if file_format == "pickle":
        with open(filename, "rb") as f:
            data_dict = pickle.load(f)

    elif file_format == "matlab":
        data_dict = {}
        loadmat(filename, mdict=data_dict)

        # fix dimensions for 0-d and 1-d data
        data_keys_0d = data_keys_0d or []
        data_keys_1d = data_keys_1d or []
        for key in data_keys_0d:
            if key in data_dict.keys():
                data_dict[key] = data_dict[key][0, 0]
        for key in data_keys_1d:
            if key in data_dict.keys():
                data_dict[key] = data_dict[key].flatten()

    elif file_format == "json":
        with open(filename) as f:
            data_dict = json.load(f)

    elif file_format == "csv":
        data_dict = pd.read_csv(filename).to_dict(orient="list")

        # fix dimensions for 0-d data
        data_keys_0d = data_keys_0d or []
        for key in data_keys_0d:
            if key in data_dict.keys():
                data_dict[key] = data_dict[key][0]

    else:
        raise ValueError(f"format '{file_format}' is not supported.")

    return data_dict
