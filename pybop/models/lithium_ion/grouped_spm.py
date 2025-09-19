import numpy as np
import pybamm
from pybamm import (
    Event,
    FunctionParameter,
    Parameter,
    ParameterValues,
    PrimaryBroadcast,
    Scalar,
    Variable,
)
from pybamm import lithium_ion as pybamm_lithium_ion
from pybamm import t as pybamm_t
from pybamm.models.full_battery_models.lithium_ion.electrode_soh import (
    get_min_max_stoichiometries,
)


class GroupedSPM(pybamm_lithium_ion.BaseModel):
    """
    A grouped parameter version of the single particle model (SPM).

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

    def __init__(self, name="Grouped Single Particle Model", **model_kwargs):
        super().__init__(name=name, **model_kwargs)

        # Unpack model options
        include_double_layer = self.options["surface form"] == "differential"

        pybamm.citations.register(
            """
            @article{Hallemans2025,
            title     = {{Physics-Based Battery Model Parametrisation from Impedance Data}},
            author    = {Hallemans, Noël and Courtier, Nicola E. and Please, Colin P. and Planden, Brady and Dhoot, Rishit and Timms, Robert and Chapman, S. Jon and Howey, David and Duncan, Stephen R.},
            journal   = {Journal of the Electrochemical Society},
            volume    = {172},
            number    = {6},
            pages     = {060507},
            year      = {2025},
            publisher = {The Electrochemical Society},
            doi       = {10.1149/1945-7111/add41b}
            }
        """
        )

        ######################
        # Variables
        ######################
        # Variables that depend on time only are created without a domain
        Q = Variable("Discharge capacity [A.h]")
        Qt = Variable("Throughput capacity [A.h]")

        # Variables that vary spatially are created with a domain
        sto_n = Variable(
            "Negative particle stoichiometry",
            domain="negative particle",
        )
        sto_p = Variable(
            "Positive particle stoichiometry",
            domain="positive particle",
        )

        # Surf takes the surface value of a variable, i.e. its boundary value on the
        # right side. This is also accessible via `boundary_value(x, "right")`, with
        # "left" providing the boundary value of the left side
        sto_n_surf = pybamm.surf(sto_n)
        sto_p_surf = pybamm.surf(sto_p)

        # Events specify points at which a solution should terminate
        self.events += [
            Event(
                "Minimum negative particle surface stoichiometry",
                pybamm.min(sto_n_surf) - 0.01,
            ),
            Event(
                "Maximum negative particle surface stoichiometry",
                (1 - 0.01) - pybamm.max(sto_n_surf),
            ),
            Event(
                "Minimum positive particle surface stoichiometry",
                pybamm.min(sto_p_surf) - 0.01,
            ),
            Event(
                "Maximum positive particle surface stoichiometry",
                (1 - 0.01) - pybamm.max(sto_p_surf),
            ),
        ]

        ######################
        # Parameters
        ######################
        # Parameters are purely symbolic at this stage, and will be set by the
        # `ParameterValues` class when the model is processed.

        F = self.param.F  # Faraday constant
        Rg = self.param.R  # Universal gas constant
        T = self.param.T_init  # Temperature
        RT_F = Rg * T / F  # Thermal voltage

        soc_init = Parameter("Initial SoC")
        x_0 = Parameter("Minimum negative stoichiometry")
        x_100 = Parameter("Maximum negative stoichiometry")
        y_100 = Parameter("Minimum positive stoichiometry")
        y_0 = Parameter("Maximum positive stoichiometry")

        # Grouped parameters
        Q_th_p = Parameter("Measured cell capacity [A.s]") / (y_0 - y_100)
        Q_th_n = Parameter("Measured cell capacity [A.s]") / (x_100 - x_0)

        tau_d_p = Parameter("Positive particle diffusion time scale [s]")
        tau_d_n = Parameter("Negative particle diffusion time scale [s]")

        tau_ct_p = Parameter("Positive electrode charge transfer time scale [s]")
        tau_ct_n = Parameter("Negative electrode charge transfer time scale [s]")

        R0 = Parameter("Series resistance [Ohm]")

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
        # Potentials
        ######################
        U_n = self.U(sto_n_surf, "negative")
        U_p = self.U(sto_p_surf, "positive")

        sto_n_init = x_0 + (x_100 - x_0) * soc_init
        sto_p_init = y_0 + (y_100 - y_0) * soc_init
        U_n_init = self.U(sto_n_init, "negative")
        U_p_init = self.U(sto_p_init, "positive")

        ######################
        # Exchange current
        ######################
        # Primary broadcasts are used to broadcast scalar quantities across a domain
        # into a vector of the right shape, for multiplying with other vectors
        alpha = 0.5  # cathodic transfer coefficient
        j0_n = sto_n_surf**alpha * (1 - sto_n_surf) ** (1 - alpha) / tau_ct_n
        j0_p = sto_p_surf**alpha * (1 - sto_p_surf) ** (1 - alpha) / tau_ct_p
        if not include_double_layer:
            # Assuming alpha = 0.5
            j_n = PrimaryBroadcast(I / (3 * Q_th_n), "negative electrode")
            j_p = PrimaryBroadcast(-I / (3 * Q_th_p), "positive electrode")
            eta_n = 2 * RT_F * pybamm.arcsinh(j_n / (2 * j0_n))
            eta_p = 2 * RT_F * pybamm.arcsinh(j_p / (2 * j0_p))
            v_s_n = pybamm.x_average(eta_n + U_n)
            v_s_p = pybamm.x_average(eta_p + U_p)

        ######################
        # Double layer
        ######################
        if include_double_layer:
            # Additional variables
            v_s_n = Variable("Negative particle surface voltage variable [V]")
            v_s_p = Variable("Positive particle surface voltage variable [V]")

            # Additional parameters
            C_p = Parameter("Positive electrode capacitance [F]")
            C_n = Parameter("Negative electrode capacitance [F]")

            # Overpotentials
            eta_n = PrimaryBroadcast(v_s_n - U_n, "negative electrode")
            eta_p = PrimaryBroadcast(v_s_p - U_p, "positive electrode")

            # Exchange current
            j_n = j0_n * (
                pybamm.exp((1 - alpha) * eta_n / RT_F)
                - pybamm.exp(-alpha * eta_n / RT_F)
            )
            j_p = j0_p * (
                pybamm.exp((1 - alpha) * eta_p / RT_F)
                - pybamm.exp(-alpha * eta_p / RT_F)
            )

            # Electrode surface potentials
            self.rhs[v_s_n] = 1 / C_n * (I - 3 * Q_th_n * pybamm.x_average(j_n))
            self.rhs[v_s_p] = 1 / C_p * (-I - 3 * Q_th_p * pybamm.x_average(j_p))
            self.initial_conditions[v_s_n] = U_n_init
            self.initial_conditions[v_s_p] = U_p_init

        ######################
        # Particles
        ######################
        # The div and grad operators will be converted to the appropriate matrix
        # multiplication at the discretisation stage
        self.rhs[sto_n] = pybamm.div(pybamm.grad(sto_n) / tau_d_n)
        self.rhs[sto_p] = pybamm.div(pybamm.grad(sto_p) / tau_d_p)

        # Boundary conditions must be provided for equations with spatial derivatives
        self.boundary_conditions[sto_n] = {
            "left": (Scalar(0), "Neumann"),
            "right": (-tau_d_n * pybamm.x_average(j_n), "Neumann"),
        }
        self.boundary_conditions[sto_p] = {
            "left": (Scalar(0), "Neumann"),
            "right": (-tau_d_p * pybamm.x_average(j_p), "Neumann"),
        }

        self.initial_conditions[sto_n] = sto_n_init
        self.initial_conditions[sto_p] = sto_p_init

        ######################
        # Cell voltage
        ######################
        V = v_s_p - v_s_n - R0 * I

        # Save the initial OCV
        self.param.ocv_init = U_p_init - U_n_init

        # Events specify points at which a solution should terminate
        self.events += [
            Event("Minimum voltage [V]", V - self.param.voltage_low_cut),
            Event("Maximum voltage [V]", self.param.voltage_high_cut - V),
        ]

        ######################
        # (Some) variables
        ######################
        # The `variables` dictionary contains all variables that might be useful for
        # visualising the solution of the model
        self.variables = {
            "Negative particle stoichiometry": sto_n,
            "Negative particle surface stoichiometry": PrimaryBroadcast(
                sto_n_surf, "negative electrode"
            ),
            "Negative particle surface voltage variable [V]": v_s_n,
            "Negative particle surface voltage [V]": PrimaryBroadcast(
                v_s_n, "negative electrode"
            ),
            "Negative electrode potential [V]": eta_n
            - pybamm.boundary_value(eta_n, "left"),
            "Positive particle stoichiometry": sto_p,
            "Positive particle surface stoichiometry": PrimaryBroadcast(
                sto_p_surf, "positive electrode"
            ),
            "Positive particle surface voltage variable [V]": v_s_p,
            "Positive particle surface voltage [V]": PrimaryBroadcast(
                v_s_p, "positive electrode"
            ),
            "Positive electrode potential [V]": V
            + eta_p
            - pybamm.boundary_value(eta_p, "right"),
            "Time [s]": pybamm_t,
            "Current [A]": I,
            "Current variable [A]": I,  # for compatibility with pybamm.Experiment
            "Discharge capacity [A.h]": Q,
            "Throughput capacity [A.h]": Qt,
            "Voltage [V]": V,
            "Battery voltage [V]": V,
            "Open-circuit voltage [V]": U_p - U_n,
        }

    def U(self, sto, domain):
        """
        Dimensional open-circuit potential [V], calculated as U(x) = U_ref(x).
        Credit: PyBaMM
        """
        # bound stoichiometry between tol and 1-tol. Adding 1/sto + 1/(sto-1) later
        # will ensure that ocp goes to +- infinity if sto goes into that region
        # anyway
        Domain = domain.capitalize()
        tol = pybamm.settings.tolerances["U__c_s"]
        sto = pybamm.maximum(pybamm.minimum(sto, 1 - tol), tol)
        inputs = {f"{Domain} particle surface stoichiometry": sto}
        u_ref = FunctionParameter(f"{Domain} electrode OCP [V]", inputs)

        # add a term to ensure that the OCP goes to infinity at 0 and -infinity at 1
        # this will not affect the OCP for most values of sto
        out = u_ref + 1e-6 * (1 / sto + 1 / (sto - 1))

        if domain == "negative":
            out.print_name = r"U_\mathrm{n}(c^\mathrm{surf}_\mathrm{s,n})"
        elif domain == "positive":
            out.print_name = r"U_\mathrm{p}(c^\mathrm{surf}_\mathrm{s,p})"
        return out

    def build_model(self):
        """
        Build model variables and equations
        Credit: PyBaMM
        """
        self._build_model()

        self._built = True
        pybamm.logger.info(f"Finish building {self.name}")

    @property
    def default_parameter_values(self) -> ParameterValues:
        param = ParameterValues("Chen2020")
        ce0 = param["Initial concentration in electrolyte [mol.m-3]"]
        T = param["Ambient temperature [K]"]
        param["Electrolyte conductivity [S.m-1]"] = param[
            "Electrolyte conductivity [S.m-1]"
        ](ce0, T)
        return self.create_grouped_parameters(param)

    @property
    def default_quick_plot_variables(self):
        return [
            "Negative particle surface stoichiometry",
            "Positive particle surface stoichiometry",
            "Current [A]",
            {
                "Negative electrode potential [V]",
                "Negative particle surface voltage [V]",
            },
            {
                "Positive electrode potential [V]",
                "Positive particle surface voltage [V]",
            },
            {"Open-circuit voltage [V]", "Voltage [V]"},
        ]

    @property
    def default_var_pts(self):
        x_n = pybamm.SpatialVariable(
            "x_n",
            domain=["negative electrode"],
            coord_sys="cartesian",
        )
        x_s = pybamm.SpatialVariable(
            "x_s",
            domain=["separator"],
            coord_sys="cartesian",
        )
        x_p = pybamm.SpatialVariable(
            "x_p",
            domain=["positive electrode"],
            coord_sys="cartesian",
        )

        # Add particle domains
        r_n = pybamm.SpatialVariable(
            "r_n",
            domain=["negative particle"],
            auxiliary_domains={"secondary": "negative electrode"},
            coord_sys="spherical polar",
        )
        r_p = pybamm.SpatialVariable(
            "r_p",
            domain=["positive particle"],
            auxiliary_domains={"secondary": "positive electrode"},
            coord_sys="spherical polar",
        )

        return {x_n: 20, x_s: 20, x_p: 20, r_n: 20, r_p: 20}

    @property
    def default_geometry(self):
        l_p = Parameter("Positive electrode relative thickness")
        l_n = Parameter("Negative electrode relative thickness")

        return {
            "negative electrode": {"x_n": {"min": 0, "max": l_n}},
            "separator": {"x_s": {"min": l_n, "max": 1 - l_p}},
            "positive electrode": {"x_p": {"min": 1 - l_p, "max": 1}},
            "negative particle": {"r_n": {"min": 0, "max": 1}},
            "positive particle": {"r_p": {"min": 0, "max": 1}},
        }

    @property
    def default_submesh_types(self):
        return {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
        }

    @property
    def default_spatial_methods(self):
        return {
            "negative electrode": pybamm.FiniteVolume(),
            "separator": pybamm.FiniteVolume(),
            "positive electrode": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
        }

    @staticmethod
    def create_grouped_parameters(parameter_values: ParameterValues) -> ParameterValues:
        """
        Create a parameter set for the Grouped Single Particle Model with Electrolyte from a
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
        F = param["Faraday constant [C.mol-1]"]
        alpha_p = param["Positive electrode active material volume fraction"]
        alpha_n = param["Negative electrode active material volume fraction"]
        c_max_p = param["Maximum concentration in positive electrode [mol.m-3]"]
        c_max_n = param["Maximum concentration in negative electrode [mol.m-3]"]
        L_p = param["Positive electrode thickness [m]"]
        L_n = param["Negative electrode thickness [m]"]
        epsilon_p = param["Positive electrode porosity"]
        epsilon_n = param["Negative electrode porosity"]
        R_p = param["Positive particle radius [m]"]
        R_n = param["Negative particle radius [m]"]
        D_p = param["Positive particle diffusivity [m2.s-1]"]
        D_n = param["Negative particle diffusivity [m2.s-1]"]
        b_p = param["Positive electrode Bruggeman coefficient (electrolyte)"]
        b_n = param["Negative electrode Bruggeman coefficient (electrolyte)"]
        Cdl_p = param["Positive electrode double-layer capacity [F.m-2]"]
        Cdl_n = param["Negative electrode double-layer capacity [F.m-2]"]
        m_p = 3.42e-6  # (A/m2)(m3/mol)**1.5
        m_n = 6.48e-7  # (A/m2)(m3/mol)**1.5
        sigma_p = (
            param["Positive electrode conductivity [S.m-1]"]
            * alpha_p ** param["Positive electrode Bruggeman coefficient (electrode)"]
        )
        sigma_n = (
            param["Negative electrode conductivity [S.m-1]"]
            * alpha_n ** param["Negative electrode Bruggeman coefficient (electrode)"]
        )

        # Separator and electrolyte properties
        ce0 = param["Initial concentration in electrolyte [mol.m-3]"]
        L_s = param["Separator thickness [m]"]
        epsilon_sep = param["Separator porosity"]
        b_sep = param["Separator Bruggeman coefficient (electrolyte)"]
        sigma_e = param["Electrolyte conductivity [S.m-1]"]  # (ce0, T)

        # Compute the cell area and thickness
        A = param["Electrode height [m]"] * param["Electrode width [m]"]
        L = L_p + L_n + L_s

        # Compute the series resistance
        Re = (
            L_p / (3 * epsilon_p**b_p)
            + L_s / (epsilon_sep**b_sep)
            + L_n / (3 * epsilon_n**b_n)
        ) / (sigma_e * A)
        Rs = (L_p / sigma_p + L_n / sigma_n) / (3 * A)
        R0 = Re + Rs + param["Contact resistance [Ohm]"]

        # Compute the stoichiometry limits and initial SOC
        x_0, x_100, y_100, y_0 = get_min_max_stoichiometries(param)
        sto_p_init = (
            param["Initial concentration in positive electrode [mol.m-3]"] / c_max_p
        )
        soc_init = (sto_p_init - y_0) / (y_100 - y_0)

        # Compute the capacity within the stoichiometry limits
        Q_th_p = F * alpha_p * c_max_p * L_p * A
        Q_th_n = F * alpha_n * c_max_n * L_n * A
        Q_meas_p = (y_0 - y_100) * Q_th_p
        Q_meas_n = (x_100 - x_0) * Q_th_n
        if abs(Q_meas_n / Q_meas_p - 1) > 1e-6:
            raise ValueError(
                "The measured capacity should be the same for both electrodes."
            )

        # Grouped parameters
        Q_meas = (Q_meas_n + Q_meas_p) / 2

        tau_d_p = R_p**2 / D_p
        tau_d_n = R_n**2 / D_n

        tau_ct_p = F * R_p / (m_p * np.sqrt(ce0))
        tau_ct_n = F * R_n / (m_n * np.sqrt(ce0))

        C_p = 3 * alpha_p * Cdl_p * L_p * A / R_p
        C_n = 3 * alpha_n * Cdl_n * L_n * A / R_n

        l_p = L_p / L
        l_n = L_n / L

        parameter_dictionary = {
            "Nominal cell capacity [A.h]": param["Nominal cell capacity [A.h]"],
            "Current function [A]": param["Current function [A]"],
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
            "Measured cell capacity [A.s]": Q_meas,
            "Positive particle diffusion time scale [s]": tau_d_p,
            "Negative particle diffusion time scale [s]": tau_d_n,
            "Positive electrode charge transfer time scale [s]": tau_ct_p,
            "Negative electrode charge transfer time scale [s]": tau_ct_n,
            "Positive electrode capacitance [F]": C_p,
            "Negative electrode capacitance [F]": C_n,
            "Positive electrode relative thickness": l_p,
            "Negative electrode relative thickness": l_n,
            "Series resistance [Ohm]": R0,
        }
        parameter_values = ParameterValues(values=parameter_dictionary)
        parameter_values._set_initial_state = set_initial_state  # noqa: SLF001
        return parameter_values


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

    if isinstance(initial_value, int | float):
        if not 0 <= initial_value <= 1:
            raise ValueError("Initial SOC should be between 0 and 1")
        parameter_values["Initial SoC"] = initial_value

    else:
        raise ValueError("Initial value must be a float between 0 and 1.")

    return parameter_values
