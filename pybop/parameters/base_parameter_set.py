# import pybamm
# import json
# import pybop

# class ParameterSet:
#     """
#     Class for creating parameter sets in PyBOP.
#     """

# def __new__(cls, method, name):
#     if method.casefold() == "pybamm":
#         return pybamm.ParameterValues(name).copy()
#     else:
#         raise ValueError("Only PyBaMM parameter sets are currently implemented")

#     def __init__(self):
#         pass

#     def import_parameters(self, json_path):
#         """
#         Import parameters from a JSON file.
#         """

#         # Read JSON file
#         with open(json_path, 'r') as file:
#             params = json.load(file)

#         # Set attributes based on the dictionary
#         for key, value in params.items():
#             if key == "Open-circuit voltage [V]":
#                 # Assuming `pybop.empirical.Thevenin().default_parameter_values` is a dictionary
#                 value = pybop.empirical.Thevenin().default_parameter_values["Open-circuit voltage [V]"]
#             setattr(self, key, value)

import json
import pybamm
import pybop


class ParameterSet:
    """
    Class for creating and importing parameter sets.
    """

    def __init__(self, json_path=None):
        self.json_path = json_path

    def import_parameters(self, json_path=None):
        """
        Import parameters from a JSON file.
        """
        if json_path is None:
            json_path = self.json_path

        # Read JSON file
        with open(json_path, "r") as file:
            params = json.load(file)

        # Set attributes based on the dictionary
        if "Open-circuit voltage [V]" in params:
            params[
                "Open-circuit voltage [V]"
            ] = pybop.empirical.Thevenin().default_parameter_values[
                "Open-circuit voltage [V]"
            ]

        return params

    @classmethod
    def pybamm(cls, name):
        """
        Create a PyBaMM parameter set.
        """
        return pybamm.ParameterValues(name).copy()
