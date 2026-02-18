import pybamm
from pybamm import FunctionParameter, Parameter
from pybamm import lithium_ion as pybamm_lithium_ion

from pybop.models.lithium_ion.utils import InverseOCV


class BaseGroupedModel(pybamm_lithium_ion.BaseModel):
    """
    A base model for PyBOP's grouped-parameter lithium-ion battery models.

    Parameters
    ----------
    name : str, optional
        The name of the model.
    **model_kwargs : optional
        Valid PyBaMM model option keys and their values, for example:
        options : dict, optional
            A dictionary of options to customise the behaviour of the PyBaMM model.
        build : bool, optional
            If True, the model is built upon creation (default: False).
    """

    def __init__(self, name="Base Model", **model_kwargs):
        super().__init__(name=name, **model_kwargs)

    def build_model(self):
        """
        Build model variables and equations
        Credit: PyBaMM
        """
        self._build_model()

        self._built = True
        pybamm.logger.info(f"Finish building {self.name}")

    @staticmethod
    def set_initial_state(
        initial_value,
        parameter_values,
        direction=None,
        param=None,
        inplace=True,
        options=None,
        inputs=None,
        tol=1e-6,
    ):
        """
        Set the value of the initial state of charge.

        Parameters
        ----------
        initial_value : float
            Target initial value.
            If float, interpreted as SOC, must be between 0 and 1.
            If string e.g. "4 V", interpreted as voltage, must be between V_min and V_max.
        parameter_values : :class:`pybamm.ParameterValues`
            Parameters and their corresponding values.
        param : :class:`pybamm.LithiumIonParameters`, optional
            The symbolic parameter set to use for the simulation.
            If not provided, the default parameter set will be used.
        inplace: bool, optional
            If True, replace the parameters values in place. Otherwise, return a new set of
            parameter values. Default is True.
        options : dict-like, optional
            A dictionary of options to be passed to the model, see
            :class:`pybamm.BatteryModelOptions`.
        inputs : dict, optional
            A dictionary of input parameters to pass to the model when solving.
        tol : float, optional
            The tolerance for the solver used to compute the initial stoichiometries.
            A lower value results in higher precision but may increase computation time.
            Default is 1e-6.
        """
        parameter_values = parameter_values if inplace else parameter_values.copy()

        if isinstance(initial_value, str) and initial_value.endswith("V"):
            V_init = float(initial_value[:-1])
            V_min = parameter_values.evaluate(
                pybamm.Parameter("Lower voltage cut-off [V]"), inputs=inputs
            )
            V_max = parameter_values.evaluate(
                pybamm.Parameter("Upper voltage cut-off [V]"), inputs=inputs
            )

            if not V_min - tol <= V_init <= V_max + tol:
                raise ValueError(
                    f"Initial voltage {V_init}V is outside the voltage limits ({V_min}, {V_max})."
                )

            x_0 = Parameter("Minimum negative stoichiometry")
            x_100 = Parameter("Maximum negative stoichiometry")
            y_100 = Parameter("Minimum positive stoichiometry")
            y_0 = Parameter("Maximum positive stoichiometry")

            def ocv_function(soc):
                sto_p = y_0 - soc * (y_0 - y_100)
                sto_n = x_0 + soc * (x_100 - x_0)
                U_p = FunctionParameter(
                    "Positive electrode OCP [V]",
                    {"Positive particle stoichiometry": sto_p},
                )
                U_n = FunctionParameter(
                    "Negative electrode OCP [V]",
                    {"Negative particle stoichiometry": sto_n},
                )

                return parameter_values.evaluate(U_p - U_n, inputs=inputs)

            inverse_ocv = InverseOCV(ocv_function)
            soc = inverse_ocv(V_init)

        elif isinstance(initial_value, int | float):
            soc = initial_value

        else:
            raise ValueError("Initial value must be a float or a string ending in 'V'.")

        if not 0 <= soc <= 1:
            raise ValueError("Initial SOC should be between 0 and 1.")

        parameter_values["Initial SoC"] = soc

        return parameter_values
