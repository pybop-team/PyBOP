import pybamm
from pybamm import (
    Event,
    FunctionParameter,
    Parameter,
    ParameterValues,
    PrimaryBroadcast,
    Scalar,
    SpatialVariable,
    Variable,
)
from pybamm import t as pybamm_t
from pybamm.models.full_battery_models.lithium_ion.electrode_soh import (
    get_min_max_stoichiometries,
)

from pybop.models.alternative_functions import FunctionalDiffusionTime
from pybop.models.lithium_ion.base_model import BaseGroupedModel


class GroupedSPMe(BaseGroupedModel):
    """
    A grouped parameter version of the single particle model with electrolyte (SPMe).

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

    def __init__(
        self, name="Grouped Single Particle Model with Electrolyte", **model_kwargs
    ):
        super().__init__(name=name, **model_kwargs)

        # Unpack model options
        include_double_layer = self.options["surface form"] == "differential"

        pybamm.citations.register("Chen2020")  # for the OCPs
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
        )  # Note that the electrode electrolyte timescales have been replaced by relative transport efficiencies

        ######################
        # Variables
        ######################
        # Variables that depend on time only are created without a domain
        Q = Variable("Discharge capacity [A.h]")
        Qt = Variable("Throughput capacity [A.h]")

        v_s_n = Variable("Negative particle surface voltage variable [V]")
        v_s_p = Variable("Positive particle surface voltage variable [V]")

        # Variables that vary spatially are created with a domain
        sto_n = Variable(
            "Negative particle stoichiometry",
            domain="negative particle",
        )
        sto_p = Variable(
            "Positive particle stoichiometry",
            domain="positive particle",
        )
        sto_e_n = Variable(
            "Negative electrode electrolyte stoichiometry",
            domain="negative electrode",
        )
        sto_e_sep = Variable(
            "Separator electrolyte stoichiometry",
            domain="separator",
        )
        sto_e_p = Variable(
            "Positive electrode electrolyte stoichiometry",
            domain="positive electrode",
        )

        # Spatial variables
        x_n = SpatialVariable("x_n", domain=["negative electrode"])
        x_p = SpatialVariable("x_p", domain=["positive electrode"])

        # Surf takes the surface value of a variable, i.e. its boundary value on the
        # right side. This is also accessible via `boundary_value(x, "right")`, with
        # "left" providing the boundary value of the left side
        sto_n_surf = pybamm.surf(sto_n)
        sto_p_surf = pybamm.surf(sto_p)

        # Events specify points at which a solution should terminate
        tol = pybamm.settings.tolerances["U__c_s"]
        self.events += [
            Event(
                "Minimum negative particle surface stoichiometry",
                pybamm.min(sto_n_surf) - tol,
            ),
            Event(
                "Maximum negative particle surface stoichiometry",
                (1 - tol) - pybamm.max(sto_n_surf),
            ),
            Event(
                "Minimum positive particle surface stoichiometry",
                pybamm.min(sto_p_surf) - tol,
            ),
            Event(
                "Maximum positive particle surface stoichiometry",
                (1 - tol) - pybamm.max(sto_p_surf),
            ),
            # model does not capture electrolyte depletion, use the DFN instead
            Event(
                "Minimum negative electrode electrolyte stoichiometry",
                pybamm.min(sto_e_n) - 0,
            ),
            Event(
                "Minimum separator electrolyte stoichiometry",
                pybamm.min(sto_e_sep) - 0,
            ),
            Event(
                "Minimum positive electrode electrolyte stoichiometry",
                pybamm.min(sto_e_p) - 0,
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
        Q_th_p = Parameter("Measured cell capacity [A.h]") * 3600 / (y_0 - y_100)
        Q_th_n = Parameter("Measured cell capacity [A.h]") * 3600 / (x_100 - x_0)
        Q_e = Parameter("Reference electrolyte capacity [A.h]") * 3600

        tau_ct_p = Parameter("Positive electrode charge transfer time scale [s]")
        tau_ct_n = Parameter("Negative electrode charge transfer time scale [s]")

        l_p = Parameter("Positive electrode relative thickness")
        l_n = Parameter("Negative electrode relative thickness")

        t_plus = Parameter("Cation transference number")

        R0 = Parameter("Series resistance [Ohm]")

        zeta_n = Parameter("Negative electrode relative porosity")
        zeta_p = Parameter("Positive electrode relative porosity")

        tau_e = Parameter("Electrolyte diffusion time scale [s]")
        beta_n = Parameter("Negative electrode relative transport efficiency")
        beta_p = Parameter("Positive electrode relative transport efficiency")

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

        eta_e = (2 * RT_F * (1 - t_plus)) * (
            pybamm.x_average(pybamm.log(sto_e_p))
            - pybamm.x_average(pybamm.log(sto_e_n))
        )

        ######################
        # Exchange current
        ######################
        # Primary broadcasts are used to broadcast scalar quantities across a domain
        # into a vector of the right shape, for multiplying with other vectors

        # Overpotentials
        eta_n = (v_s_n - U_n) + (2 * RT_F * (1 - t_plus)) * (
            pybamm.x_average(pybamm.log(sto_e_n)) - pybamm.log(sto_e_n)
        )
        eta_p = (v_s_p - U_p) + (2 * RT_F * (1 - t_plus)) * (
            pybamm.x_average(pybamm.log(sto_e_p)) - pybamm.log(sto_e_p)
        )

        # Exchange rates
        j_n = self.j(sto_n_surf, sto_e_n, eta_n / RT_F, "negative") / tau_ct_n
        j_p = self.j(sto_p_surf, sto_e_p, eta_p / RT_F, "positive") / tau_ct_p

        ######################
        # Double layer
        ######################
        if include_double_layer:
            # Additional parameters
            C_p = Parameter("Positive electrode capacitance [F]")
            C_n = Parameter("Negative electrode capacitance [F]")

            # Electrode surface potentials
            self.rhs[v_s_n] = (I - 3 * Q_th_n * pybamm.x_average(j_n)) / C_n
            self.rhs[v_s_p] = (-I - 3 * Q_th_p * pybamm.x_average(j_p)) / C_p
        else:
            self.algebraic[v_s_n] = I - 3 * Q_th_n * pybamm.x_average(j_n)
            self.algebraic[v_s_p] = -I - 3 * Q_th_p * pybamm.x_average(j_p)

        self.initial_conditions[v_s_n] = U_n_init
        self.initial_conditions[v_s_p] = U_p_init

        ######################
        # Particles
        ######################
        # The div and grad operators will be converted to the appropriate matrix
        # multiplication at the discretisation stage
        self.rhs[sto_n] = pybamm.div(
            pybamm.grad(sto_n) / self.tau_d(sto_n, T, "negative")
        )
        self.rhs[sto_p] = pybamm.div(
            pybamm.grad(sto_p) / self.tau_d(sto_p, T, "positive")
        )

        # Boundary conditions must be provided for equations with spatial derivatives
        self.boundary_conditions[sto_n] = {
            "left": (Scalar(0), "Neumann"),
            "right": (
                -self.tau_d(sto_n_surf, T, "negative") * pybamm.x_average(j_n),
                "Neumann",
            ),
        }
        self.boundary_conditions[sto_p] = {
            "left": (Scalar(0), "Neumann"),
            "right": (
                -self.tau_d(sto_p_surf, T, "positive") * pybamm.x_average(j_p),
                "Neumann",
            ),
        }

        self.initial_conditions[sto_n] = sto_n_init
        self.initial_conditions[sto_p] = sto_p_init

        ######################
        # Electrolyte
        ######################
        self.rhs[sto_e_n] = (
            pybamm.div(
                pybamm.grad(sto_e_n) * beta_n / tau_e - (t_plus * I / Q_e) * x_n / l_n
            )
            + (3 / Q_e) * Q_th_n * j_n / l_n
        ) / zeta_n
        self.rhs[sto_e_sep] = pybamm.div(
            pybamm.grad(sto_e_sep) / tau_e - t_plus * I / Q_e
        )
        self.rhs[sto_e_p] = (
            pybamm.div(
                pybamm.grad(sto_e_p) * beta_p / tau_e
                - (t_plus * I / Q_e) * (1 - x_p) / l_p
            )
            + (3 / Q_e) * Q_th_p * j_p / l_p
        ) / zeta_p

        self.boundary_conditions[sto_e_n] = {
            "left": (Scalar(0), "Neumann"),
            "right": (pybamm.boundary_gradient(sto_e_sep, "left") / beta_n, "Neumann"),
        }
        self.boundary_conditions[sto_e_sep] = {
            "left": (pybamm.boundary_value(sto_e_n, "right"), "Dirichlet"),
            "right": (pybamm.boundary_value(sto_e_p, "left"), "Dirichlet"),
        }
        self.boundary_conditions[sto_e_p] = {
            "left": (pybamm.boundary_gradient(sto_e_sep, "right") / beta_p, "Neumann"),
            "right": (Scalar(0), "Neumann"),
        }

        self.initial_conditions[sto_e_n] = Scalar(1)
        self.initial_conditions[sto_e_sep] = Scalar(1)
        self.initial_conditions[sto_e_p] = Scalar(1)

        ######################
        # Cell voltage
        ######################
        V = v_s_p - v_s_n + eta_e - R0 * I

        # Save the initial OCV
        self.param.ocv_init = U_p_init - U_n_init

        # Events specify points at which a solution should terminate
        self.events += [
            Event("Minimum voltage [V]", V - self.param.voltage_low_cut),
            Event("Maximum voltage [V]", self.param.voltage_high_cut - V),
        ]

        ######################
        # Voltage components
        ######################
        # Include the following variables to enable plotting via PyBaMM's plot_voltage_components
        ocp_n_bulk = self.U(
            pybamm.x_average(Q_th_n * pybamm.r_average(sto_n))
            / pybamm.x_average(Q_th_n),
            "negative",
        )
        ocp_p_bulk = self.U(
            pybamm.x_average(Q_th_p * pybamm.r_average(sto_p))
            / pybamm.x_average(Q_th_p),
            "positive",
        )
        voltage_components = {
            "Battery voltage [V]": V,
            "Battery open-circuit voltage [V]": ocp_p_bulk - ocp_n_bulk,
            "Battery particle concentration overpotential [V]": (
                (pybamm.x_average(self.U(sto_p_surf, "positive")) - ocp_p_bulk)
                - (pybamm.x_average(self.U(sto_n_surf, "negative")) - ocp_n_bulk)
            ),
            "X-averaged battery reaction overpotential [V]": pybamm.x_average(eta_p)
            - pybamm.x_average(eta_n),
            "X-averaged battery concentration overpotential [V]": eta_e,
            "X-averaged battery electrolyte ohmic losses [V]": Scalar(0),
            "X-averaged battery solid phase ohmic losses [V]": Scalar(0),
            "Contact overpotential [V]": R0 * I,  #  includes Ohmic losses in this model
            # and split by electrode
            "Negative electrode bulk open-circuit potential [V]": ocp_n_bulk,
            "Positive electrode bulk open-circuit potential [V]": ocp_p_bulk,
            "Negative particle concentration overpotential [V]": pybamm.x_average(
                self.U(sto_n_surf, "negative")
            )
            - ocp_n_bulk,
            "Positive particle concentration overpotential [V]": pybamm.x_average(
                self.U(sto_p_surf, "positive")
            )
            - ocp_p_bulk,
            "X-averaged negative electrode reaction overpotential [V]"
            "": pybamm.x_average(eta_n),
            "X-averaged positive electrode reaction overpotential [V]"
            "": pybamm.x_average(eta_p),
            "X-averaged battery negative solid phase ohmic losses [V]": Scalar(0),
            "X-averaged battery positive solid phase ohmic losses [V]": Scalar(0),
        }

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
            "Negative electrode electrolyte stoichiometry": sto_e_n,
            "Separator electrolyte stoichiometry": sto_e_sep,
            "Positive electrode electrolyte stoichiometry": sto_e_p,
            "Electrolyte stoichiometry": pybamm.concatenation(
                sto_e_n, sto_e_sep, sto_e_p
            ),
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
            "Electrolyte potential [V]": -v_s_n
            - (2 * RT_F * (1 - t_plus))
            * (
                pybamm.boundary_value(pybamm.log(sto_e_n), "left")
                - pybamm.log(pybamm.concatenation(sto_e_n, sto_e_sep, sto_e_p))
            ),
            "Time [s]": pybamm_t,
            "Time [h]": pybamm_t / 3600,
            "Current [A]": I,
            "Current variable [A]": I,  # for compatibility with pybamm.Experiment
            "Discharge capacity [A.h]": Q,
            "Throughput capacity [A.h]": Qt,
            "Voltage [V]": V,
            "Voltage expression [V]": V,  # for compatibility with "voltage as a state"
            "Open-circuit voltage [V]": U_p - U_n,
            **voltage_components,
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

    def tau_d(self, sto, T, domain):
        """
        Dimensional solid-state diffusion time scale [s].
        """
        Domain = domain.capitalize()
        inputs = {f"{Domain} particle surface stoichiometry": sto, "Temperature [K]": T}
        return FunctionParameter(f"{Domain} particle diffusion time scale [s]", inputs)

    def j(self, sto_surf, sto_e, eta_RT_F, domain):
        """
        Dimensionless exchange rate.
        """
        Domain = domain.capitalize()
        inputs = {
            f"{Domain} particle surface stoichiometry": sto_surf,
            f"{Domain} electrode electrolyte stoichiometry": sto_e,
            f"{Domain} electrode dimensionless overpotential": eta_RT_F,
        }
        return FunctionParameter(
            f"{Domain} electrode dimensionless exchange rate", inputs
        )

    @property
    def default_parameter_values(self) -> ParameterValues:
        param = ParameterValues("Chen2020")
        ce0 = param["Initial concentration in electrolyte [mol.m-3]"]
        T = param["Ambient temperature [K]"]
        param["Electrolyte conductivity [S.m-1]"] = param[
            "Electrolyte conductivity [S.m-1]"
        ](ce0, T)
        param["Electrolyte diffusivity [m2.s-1]"] = param[
            "Electrolyte diffusivity [m2.s-1]"
        ](ce0, T)
        return self.create_grouped_parameters(param)

    @property
    def default_quick_plot_variables(self):
        return [
            "Negative particle stoichiometry",
            "Electrolyte stoichiometry",
            "Positive particle stoichiometry",
            "Current [A]",
            {
                "Negative electrode potential [V]",
                "Negative particle surface voltage [V]",
            },
            "Electrolyte potential [V]",
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
        F = pybamm.constants.F.value
        T = param["Ambient temperature [K]"]
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
        De = param["Electrolyte diffusivity [m2.s-1]"]  # (ce0, T)
        L_s = param["Separator thickness [m]"]
        epsilon_sep = param["Separator porosity"]
        b_sep = param["Separator Bruggeman coefficient (electrolyte)"]
        t_plus = param["Cation transference number"]
        kappa_e = param["Electrolyte conductivity [S.m-1]"]  # (ce0, T)

        # Get reference exchange current density [A.m-2]
        j0_p = param.evaluate(
            param["Positive electrode exchange-current density [A.m-2]"](
                ce0, c_max_p / 2, c_max_p, T
            )
        )
        j0_n = param.evaluate(
            param["Negative electrode exchange-current density [A.m-2]"](
                ce0, c_max_n / 2, c_max_n, T
            )
        )

        # Compute the cell area and thickness
        A = param["Electrode height [m]"] * param["Electrode width [m]"]
        L = L_p + L_n + L_s

        # Compute the series resistance
        Re = (
            L_p / (3 * epsilon_p**b_p)
            + L_s / (epsilon_sep**b_sep)
            + L_n / (3 * epsilon_n**b_n)
        ) / (kappa_e * A)
        Rs = (L_p / sigma_p + L_n / sigma_n) / (3 * A)
        R0 = Re + Rs + param["Contact resistance [Ohm]"]

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
        Q_e = F * epsilon_sep * ce0 * L * A / 3600

        zeta_p = epsilon_p / epsilon_sep
        zeta_n = epsilon_n / epsilon_sep

        try:
            tau_d_p = R_p**2 / D_p
        except TypeError:
            tau_d_p = FunctionalDiffusionTime(R_p**2, D_p, c_max_p)

        try:
            tau_d_n = R_n**2 / D_n
        except TypeError:
            tau_d_n = FunctionalDiffusionTime(R_n**2, D_n, c_max_n)

        tau_e = epsilon_sep * L**2 / (epsilon_sep**b_sep * De)
        beta_p = epsilon_p**b_p / epsilon_sep**b_sep
        beta_n = epsilon_n**b_n / epsilon_sep**b_sep

        tau_ct_p = c_max_p * F * R_p / (2 * j0_p)
        tau_ct_n = c_max_n * F * R_n / (2 * j0_n)

        C_p = 3 * alpha_p * Cdl_p * L_p * A / R_p
        C_n = 3 * alpha_n * Cdl_n * L_n * A / R_n

        l_p = L_p / L
        l_n = L_n / L

        parameter_dictionary = {
            "Nominal cell capacity [A.h]": param["Nominal cell capacity [A.h]"],
            "Current function [A]": param["Current function [A]"],
            "Ambient temperature [K]": T,
            "Initial temperature [K]": T,
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
            "Reference electrolyte capacity [A.h]": Q_e,
            "Positive electrode relative porosity": zeta_p,
            "Negative electrode relative porosity": zeta_n,
            "Positive particle diffusion time scale [s]": tau_d_p,
            "Negative particle diffusion time scale [s]": tau_d_n,
            "Electrolyte diffusion time scale [s]": tau_e,
            "Positive electrode relative transport efficiency": beta_p,
            "Negative electrode relative transport efficiency": beta_n,
            "Positive electrode dimensionless exchange rate": GroupedSPMe.symmetric_butler_volmer,
            "Negative electrode dimensionless exchange rate": GroupedSPMe.symmetric_butler_volmer,
            "Positive electrode charge transfer time scale [s]": tau_ct_p,
            "Negative electrode charge transfer time scale [s]": tau_ct_n,
            "Positive electrode capacitance [F]": C_p,
            "Negative electrode capacitance [F]": C_n,
            "Cation transference number": t_plus,
            "Positive electrode relative thickness": l_p,
            "Negative electrode relative thickness": l_n,
            "Series resistance [Ohm]": R0,
        }
        parameter_values = ParameterValues(values=parameter_dictionary)
        parameter_values._set_initial_state = GroupedSPMe.set_initial_state  # noqa: SLF001
        return parameter_values
