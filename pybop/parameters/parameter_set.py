import json
import types
import pybamm
import pybop


class ParameterSet:
    """
    A class to manage the import and export of parameter sets for battery models.

    Attributes:
        json_path (str): The file path to a JSON file containing parameter data.
        params (dict): A dictionary containing parameter key-value pairs.
    """

    def __init__(self, json_path=None, params_dict=None):
        self.json_path = json_path
        self.params = params_dict or {}
        self.chemistry = None

    def import_parameters(self, json_path=None):
        """
        Import parameters from a JSON file.
        """

        # Read JSON file
        if not self.params and self.json_path:
            with open(self.json_path, "r") as file:
                self.params = json.load(file)
                self._handle_special_cases()
        if self.params["chemistry"] is not None:
            self.chemistry = self.params["chemistry"]
        return self.params

    def _handle_special_cases(self):
        """
        Handles special cases for parameter values that require custom logic.
        """
        if (
            "Open-circuit voltage [V]" in self.params
            and self.params["Open-circuit voltage [V]"] == "default"
        ):
            self.params[
                "Open-circuit voltage [V]"
            ] = pybop.empirical.Thevenin().default_parameter_values[
                "Open-circuit voltage [V]"
            ]

    def export_parameters(self, output_json_path, fit_params=None):
        """
        Export parameters to a JSON file.
        """
        if not self.params:
            raise ValueError("No parameters to export. Please import parameters first.")

        # Prepare a copy of the params to avoid modifying the original dict
        exportable_params = {**{"chemistry": self.chemistry}, **self.params.copy()}

        # Update parameter set
        if fit_params is not None:
            for i, param in enumerate(fit_params):
                exportable_params.update({param.name: param.value})

        # Replace non-serializable values
        for key, value in exportable_params.items():
            if isinstance(value, types.FunctionType) or not self.is_json_serializable(
                value
            ):
                exportable_params[key] = "Unable to write value to JSON file"

        # Write parameters to JSON file
        with open(output_json_path, "w") as file:
            json.dump(exportable_params, file, indent=4)

    def is_json_serializable(self, value):
        """
        Check if the value is serializable to JSON.
        """
        try:
            json.dumps(value)
            return True
        except (TypeError, OverflowError):
            return False

    @classmethod
    def pybamm(cls, name):
        """
        Create a PyBaMM parameter set.
        """
        return pybamm.ParameterValues(name).copy()
