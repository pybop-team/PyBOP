import pybamm
from pybamm import (
    FunctionParameter,
    Parameter,
    ParameterValues,
    Scalar,
    Variable,
)
from pybamm import t as pybamm_t
from pybamm.models.full_battery_models.lithium_ion.electrode_soh import (
    get_min_max_stoichiometries,
)

from pybop.models.lithium_ion.base_model import BaseGroupedModel


class CellTemperature(BaseGroupedModel):
    """
    A lumped thermal model for a battery cell.

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

    def __init__(self, name="Cell Temperature Model", **model_kwargs):
        super().__init__(name=name, **model_kwargs)

        # Voltage is an input to this model, not a state
        self.options["voltage as a state"] = "false"

        ######################
        # Variables
        ######################
        # Variables that depend on time only are created without a domain
        Q = Variable("Discharge capacity [A.h]")
        Qt = Variable("Throughput capacity [A.h]")

        T_cell = Variable("Cell temperature [K]")
        T_surf = Variable("Surface temperature [K]")

        ######################
        # Parameters
        ######################
        # Parameters are purely symbolic at this stage, and will be set by the
        # `ParameterValues` class when the model is processed.

        Q_meas = Parameter("Measured cell capacity [A.h]") * 3600

        soc_init = Parameter("Initial SoC")
        x_0 = Parameter("Minimum negative stoichiometry")
        x_100 = Parameter("Maximum negative stoichiometry")
        y_100 = Parameter("Minimum positive stoichiometry")
        y_0 = Parameter("Maximum positive stoichiometry")

        T_init = Parameter("Initial temperature [K]")
        T_ref = Parameter("Reference temperature [K]")

        S_n = Parameter("Negative electrode OCP entropic change [V.K-1]")
        S_p = Parameter("Positive electrode OCP entropic change [V.K-1]")

        c_th_c = Parameter("Cell thermal mass [J/K]")
        c_th_s = Parameter("Surface thermal mass [J/K]")
        h_c = Parameter("Cell heat transfer coefficient [W/K]")
        h_s = Parameter("Surface heat transfer coefficient [W/K]")

        ######################
        # Input current (positive on discharge)
        ######################
        I = self.param.current_with_time

        ######################
        # State of Charge
        ######################
        # The `rhs` dictionary contains differential equations, with the key being the
        # variable in the d/dt
        self.rhs[Q] = I / 3600
        self.rhs[Qt] = abs(I) / 3600
        # Initial conditions must be provided for the ODEs
        self.initial_conditions[Q] = Scalar(0)
        self.initial_conditions[Qt] = Scalar(0)

        ######################
        # Output voltage (as an input to the model)
        ######################
        V = pybamm.FunctionParameter("Voltage function [V]", {"Time [s]": pybamm.t})

        ######################
        # Thermal model
        ######################
        # Open-circuit voltage
        soc = soc_init - Q * 3600 / Q_meas
        sto_n = x_0 + (x_100 - x_0) * soc
        sto_p = y_0 + (y_100 - y_0) * soc
        U = (
            self.U(sto_p, "positive")
            + S_p * (T_cell - T_ref)
            - self.U(sto_n, "negative")
            - S_n * (T_cell - T_ref)
        )

        # Irreversible heating
        Q_irr = -I * (V - U)

        # Reversible heating
        dUdT = S_p - S_n
        Q_rev = -I * T_cell * dUdT

        # Ambient temperature
        T_amb = pybamm.FunctionParameter(
            "Ambient temperature [K]", {"Time [s]": pybamm.t}
        )

        # Cell temperature
        self.rhs[T_cell] = (Q_irr + Q_rev - h_c * (T_cell - T_surf)) / c_th_c
        self.rhs[T_surf] = (h_c * (T_cell - T_surf) - h_s * (T_surf - T_amb)) / c_th_s
        self.initial_conditions[T_cell] = T_init
        self.initial_conditions[T_surf] = T_init

        ######################
        # (Some) variables
        ######################
        self.variables = {
            "Negative particle stoichiometry": sto_n,
            "Positive particle stoichiometry": sto_p,
            "SoC": soc,
            "Time [s]": pybamm_t,
            "Current [A]": I,
            "Current variable [A]": I,  # for compatibility with pybamm.Experiment
            "Discharge capacity [A.h]": Q,
            "Throughput capacity [A.h]": Qt,
            "Voltage [V]": V,
            "Open-circuit voltage [V]": U,
            "Ambient temperature [K]": T_amb,
            "Cell temperature [K]": T_cell,
            "Surface temperature [K]": T_surf,
        }

    def U(self, sto, domain):
        """
        Dimensional open-circuit potential [V].
        Credit: PyBaMM
        """
        Domain = domain.capitalize()
        inputs = {f"{Domain} particle surface stoichiometry": sto}
        out = FunctionParameter(f"{Domain} electrode OCP [V]", inputs)

        if domain == "negative":
            out.print_name = r"U_\mathrm{n}(c^\mathrm{surf}_\mathrm{s,n})"
        elif domain == "positive":
            out.print_name = r"U_\mathrm{p}(c^\mathrm{surf}_\mathrm{s,p})"
        return out

    @property
    def default_parameter_values(self) -> ParameterValues:
        param = ParameterValues("Chen2020")
        param.update(
            {
                "Voltage function [V]": param.evaluate(self.param.ocv_init),
                "Cell thermal mass [J/K]": 20,
                "Surface thermal mass [J/K]": 20,
                "Cell heat transfer coefficient [W/K]": 0.05,
                "Surface heat transfer coefficient [W/K]": 0.05,
            },
            check_already_exists=False,
        )
        return self.create_grouped_parameters(param)

    @property
    def default_quick_plot_variables(self):
        return [
            "Negative particle stoichiometry",
            "Positive particle stoichiometry",
            "Current [A]",
            "SoC",
            {"Open-circuit voltage [V]", "Voltage [V]"},
            {
                "Ambient temperature [K]",
                "Cell temperature [K]",
                "Surface temperature [K]",
            },
        ]

    @property
    def default_var_pts(self):
        return {}

    @property
    def default_geometry(self):
        return {}

    @property
    def default_submesh_types(self):
        return {}

    @property
    def default_spatial_methods(self):
        return {}

    @staticmethod
    def create_grouped_parameters(parameter_values: ParameterValues) -> ParameterValues:
        """
        Create a parameter set for the Cell Temperature Model from a
        PyBaMM lithium-ion ParameterValues object.

        Parameters
        ----------
        parameter_values : pybamm.ParameterValues
            Parameters and their corresponding values.

        Returns
        -------
        parameter_values : pybamm.ParameterValues
            A new set of parameters and their values.
        """
        param = parameter_values

        # Unpack physical parameters
        F = pybamm.constants.F.value
        alpha_p = param["Positive electrode active material volume fraction"]
        alpha_n = param["Negative electrode active material volume fraction"]
        c_max_p = param["Maximum concentration in positive electrode [mol.m-3]"]
        c_max_n = param["Maximum concentration in negative electrode [mol.m-3]"]
        L_p = param["Positive electrode thickness [m]"]
        L_n = param["Negative electrode thickness [m]"]

        # Compute the cell area
        A = param["Electrode height [m]"] * param["Electrode width [m]"]

        # Compute the stoichiometry limits and initial SOC
        x_0, x_100, y_100, y_0 = get_min_max_stoichiometries(param)
        sto_p_init = (
            param["Initial concentration in positive electrode [mol.m-3]"] / c_max_p
        )
        soc_init = (sto_p_init - y_0) / (y_100 - y_0)

        # Compute the capacity within the stoichiometry limits
        Q_th_p = F * alpha_p * c_max_p * L_p * A / 3600
        Q_th_n = F * alpha_n * c_max_n * L_n * A / 3600
        Q_meas_p = (y_0 - y_100) * Q_th_p
        Q_meas_n = (x_100 - x_0) * Q_th_n
        if abs(Q_meas_n / Q_meas_p - 1) > 1e-6:
            raise ValueError(
                "The measured capacity should be the same for both electrodes."
            )

        # Grouped parameters
        Q_meas = (Q_meas_n + Q_meas_p) / 2

        parameter_dictionary = {
            "Nominal cell capacity [A.h]": param["Nominal cell capacity [A.h]"],
            "Current function [A]": param["Current function [A]"],
            "Voltage function [V]": param["Voltage function [V]"],
            "Reference temperature [K]": param["Reference temperature [K]"],
            "Ambient temperature [K]": param["Ambient temperature [K]"],
            "Initial temperature [K]": param["Ambient temperature [K]"],
            "Initial SoC": soc_init,
            "Minimum negative stoichiometry": x_0,
            "Maximum negative stoichiometry": x_100,
            "Minimum positive stoichiometry": y_100,
            "Maximum positive stoichiometry": y_0,
            "Lower voltage cut-off [V]": param["Lower voltage cut-off [V]"],
            "Upper voltage cut-off [V]": param["Upper voltage cut-off [V]"],
            "Positive electrode OCP [V]": param["Positive electrode OCP [V]"],
            "Negative electrode OCP [V]": param["Negative electrode OCP [V]"],
            "Measured cell capacity [A.h]": Q_meas,
            "Negative electrode OCP entropic change [V.K-1]": param[
                "Negative electrode OCP entropic change [V.K-1]"
            ],
            "Positive electrode OCP entropic change [V.K-1]": param[
                "Positive electrode OCP entropic change [V.K-1]"
            ],
            "Cell thermal mass [J/K]": param["Cell thermal mass [J/K]"],
            "Surface thermal mass [J/K]": param["Surface thermal mass [J/K]"],
            "Cell heat transfer coefficient [W/K]": param[
                "Cell heat transfer coefficient [W/K]"
            ],
            "Surface heat transfer coefficient [W/K]": param[
                "Surface heat transfer coefficient [W/K]"
            ],
        }
        parameter_values = ParameterValues(values=parameter_dictionary)
        parameter_values._set_initial_state = CellTemperature.set_initial_state  # noqa: SLF001
        return parameter_values
