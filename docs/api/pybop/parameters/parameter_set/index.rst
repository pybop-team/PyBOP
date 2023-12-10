:py:mod:`pybop.parameters.parameter_set`
========================================

.. py:module:: pybop.parameters.parameter_set


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pybop.parameters.parameter_set.ParameterSet




.. py:class:: ParameterSet(json_path=None, params_dict=None)


   Handles the import and export of parameter sets for battery models.

   This class provides methods to load parameters from a JSON file and to export them
   back to a JSON file. It also includes custom logic to handle special cases, such
   as parameter values that require specific initialization.

   :param json_path: Path to a JSON file containing parameter data. If provided, parameters will be imported from this file during initialization.
   :type json_path: str, optional
   :param params_dict: A dictionary of parameters to initialize the ParameterSet with. If not provided, an empty dictionary is used.
   :type params_dict: dict, optional

   .. py:method:: _handle_special_cases()

      Processes special cases for parameter values that require custom handling.

      For example, if the open-circuit voltage is specified as 'default', it will
      fetch the default value from the PyBaMM empirical Thevenin model.


   .. py:method:: export_parameters(output_json_path, fit_params=None)

      Exports parameters to a JSON file specified by `output_json_path`.

      The current state of the `params` attribute is written to the file. If `fit_params`
      is provided, these parameters are updated before export. Non-serializable values
      are handled and noted in the output JSON.

      :param output_json_path: The file path where the JSON output will be saved.
      :type output_json_path: str
      :param fit_params: Parameters that have been fitted and need to be included in the export.
      :type fit_params: list of fitted parameter objects, optional

      :raises ValueError: If there are no parameters to export.


   .. py:method:: import_parameters(json_path=None)

      Imports parameters from a JSON file specified by the `json_path` attribute.

      If a `json_path` is provided at initialization or as an argument, that JSON file
      is loaded and the parameters are stored in the `params` attribute. Special cases
      are handled appropriately.

      :param json_path: Path to the JSON file from which to import parameters. If provided, it overrides the instance's `json_path`.
      :type json_path: str, optional

      :returns: The dictionary containing the imported parameters.
      :rtype: dict

      :raises FileNotFoundError: If the specified JSON file cannot be found.


   .. py:method:: is_json_serializable(value)

      Determines if the given `value` can be serialized to JSON format.

      :param value: The value to check for JSON serializability.
      :type value: any

      :returns: True if the value is JSON serializable, False otherwise.
      :rtype: bool


   .. py:method:: pybamm(name)
      :classmethod:

      Retrieves a PyBaMM parameter set by name.

      :param name: The name of the PyBaMM parameter set to retrieve.
      :type name: str

      :returns: A PyBaMM parameter set corresponding to the provided name.
      :rtype: pybamm.ParameterValues
