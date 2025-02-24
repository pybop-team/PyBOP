import warnings

import numpy as np
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
from pybamm import lithium_ion as pybamm_lithium_ion
from pybamm import t as pybamm_t
from pybamm.input.parameters.lithium_ion.Chen2020 import (
    graphite_LGM50_ocp_Chen2020,
    nmc_LGM50_ocp_Chen2020,
)
from pybamm.models.full_battery_models.lithium_ion.electrode_soh import (
    get_min_max_stoichiometries,
)

from pybop import ParameterSet


class BaseGroupedSPMe(pybamm_lithium_ion.BaseModel):
    """
    A grouped parameter version of the single particle model with electrolyte (SPMe).

    Parameters
    ----------
    name : str, optional
        The name of the model.
    eis : bool, optional
        A flag to build the forward model for EIS predictions. Defaults to False.
    **model_kwargs : optional
        Valid PyBaMM model option keys and their values, for example:
        parameter_set : pybamm.ParameterValues or dict, optional
            The parameters for the model. If None, default parameters provided by PyBaMM are used.
        geometry : dict, optional
            The geometry definitions for the model. If None, default geometry from PyBaMM is used.
        submesh_types : dict, optional
            The types of submeshes to use. If None, default submesh types from PyBaMM are used.
        var_pts : dict, optional
            The discretization points for each variable in the model. If None, default points from PyBaMM are used.
        spatial_methods : dict, optional
            The spatial methods used for discretization. If None, default spatial methods from PyBaMM are used.
        solver : pybamm.Solver, optional
            The solver to use for simulating the model. If None, the default solver from PyBaMM is used.
        build : bool, optional
            If True, the model is built upon creation (default: False).
        options : dict, optional
            A dictionary of options to customise the behaviour of the PyBaMM model.
    """

    def __init__(
        self, name="Grouped Single Particle Model with Electrolyte", **model_kwargs
    ):
        unused_keys = []
        for key in model_kwargs.keys():
            if key not in ["build", "parameter_set", "options"]:
                unused_keys.append(key)
        if model_kwargs.get("build", True) is False:
            unused_keys.append("build")
        options = {"surface form": "false", "contact resistance": "true"}
        if model_kwargs.get("options", None) is not None:
            for key, value in model_kwargs["options"].items():
                if key in ["surface form", "contact resistance"]:
                    options[key] = value
                else:
                    unused_keys.append("options[" + key + "]")
        if any(unused_keys):
            unused_kwargs_warning = f"The input model_kwargs {unused_keys} are not currently used by the GroupedSPMe."
            warnings.warn(unused_kwargs_warning, UserWarning, stacklevel=2)

        super().__init__(options=options, name=name, build=True)

        # Unpack model options
        include_double_layer = self.options["surface form"] == "differential"

        pybamm.citations.register("Chen2020")  # for the OCPs
        pybamm.citations.register(
            """
            @article{HallemansPreprint,
            title={{Hallemans Preprint}},
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
        Q_e = Parameter("Reference electrolyte capacity [A.s]")

        tau_d_p = Parameter("Positive particle diffusion time scale [s]")
        tau_d_n = Parameter("Negative particle diffusion time scale [s]")

        tau_ct_p = Parameter("Positive electrode charge transfer time scale [s]")
        tau_ct_n = Parameter("Negative electrode charge transfer time scale [s]")

        l_p = Parameter("Positive electrode relative thickness")
        l_n = Parameter("Negative electrode relative thickness")

        t_plus = Parameter("Cation transference number")

        R0 = Parameter("Series resistance [Ohm]")

        zeta_n = Parameter("Negative electrode relative porosity")
        zeta_p = Parameter("Positive electrode relative porosity")

        tau_e_n = Parameter("Negative electrode electrolyte diffusion time scale [s]")
        tau_e_sep = Parameter("Separator electrolyte diffusion time scale [s]")
        tau_e_p = Parameter("Positive electrode electrolyte diffusion time scale [s]")

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
        alpha = 0.5  # cathodic transfer coefficient
        j0_n = (
            sto_n_surf**alpha * (sto_e_n * (1 - sto_n_surf)) ** (1 - alpha) / tau_ct_n
        )
        j0_p = (
            sto_p_surf**alpha * (sto_e_p * (1 - sto_p_surf)) ** (1 - alpha) / tau_ct_p
        )
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
            eta_n = (v_s_n - U_n) + (2 * RT_F * (1 - t_plus)) * (
                pybamm.x_average(pybamm.log(sto_e_n)) - pybamm.log(sto_e_n)
            )
            eta_p = (v_s_p - U_p) + (2 * RT_F * (1 - t_plus)) * (
                pybamm.x_average(pybamm.log(sto_e_p)) - pybamm.log(sto_e_p)
            )

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
        # Electrolyte
        ######################
        self.rhs[sto_e_n] = (
            pybamm.div(pybamm.grad(sto_e_n) / tau_e_n - (t_plus * I / Q_e) * x_n / l_n)
            + (3 / Q_e) * Q_th_n * j_n / l_n
        ) / zeta_n
        self.rhs[sto_e_sep] = pybamm.div(
            pybamm.grad(sto_e_sep) / tau_e_sep - t_plus * I / Q_e
        )
        self.rhs[sto_e_p] = (
            pybamm.div(
                pybamm.grad(sto_e_p) / tau_e_p - (t_plus * I / Q_e) * (1 - x_p) / l_p
            )
            + (3 / Q_e) * Q_th_p * j_p / l_p
        ) / zeta_p

        self.boundary_conditions[sto_e_n] = {
            "left": (Scalar(0), "Neumann"),
            "right": (
                tau_e_n * pybamm.boundary_gradient(sto_e_sep, "left") / tau_e_sep,
                "Neumann",
            ),
        }
        self.boundary_conditions[sto_e_sep] = {
            "left": (pybamm.boundary_value(sto_e_n, "right"), "Dirichlet"),
            "right": (pybamm.boundary_value(sto_e_p, "left"), "Dirichlet"),
        }
        self.boundary_conditions[sto_e_p] = {
            "left": (
                tau_e_p * pybamm.boundary_gradient(sto_e_sep, "right") / tau_e_sep,
                "Neumann",
            ),
            "right": (Scalar(0), "Neumann"),
        }

        self.initial_conditions[sto_e_n] = PrimaryBroadcast(
            Scalar(1), "negative electrode"
        )
        self.initial_conditions[sto_e_sep] = PrimaryBroadcast(Scalar(1), "separator")
        self.initial_conditions[sto_e_p] = PrimaryBroadcast(
            Scalar(1), "positive electrode"
        )

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
    def default_parameter_values(self):
        parameter_dictionary = {
            "Nominal cell capacity [A.h]": 3,
            "Current function [A]": 3,
            "Initial temperature [K]": 298.15,
            "Initial SoC": 0.5,
            "Minimum negative stoichiometry": 0.026,
            "Maximum negative stoichiometry": 0.911,
            "Minimum positive stoichiometry": 0.264,
            "Maximum positive stoichiometry": 0.854,
            "Lower voltage cut-off [V]": 2.5,
            "Upper voltage cut-off [V]": 4.2,
            "Positive electrode OCP [V]": nmc_LGM50_ocp_Chen2020,
            "Negative electrode OCP [V]": graphite_LGM50_ocp_Chen2020,
            "Measured cell capacity [A.s]": 3000,
            "Reference electrolyte capacity [A.s]": 1000,
            "Positive electrode relative porosity": 1,
            "Negative electrode relative porosity": 1,
            "Positive particle diffusion time scale [s]": 2000,
            "Negative particle diffusion time scale [s]": 2000,
            "Positive electrode electrolyte diffusion time scale [s]": 300,
            "Negative electrode electrolyte diffusion time scale [s]": 300,
            "Separator electrolyte diffusion time scale [s]": 300,
            "Positive electrode charge transfer time scale [s]": 500,
            "Negative electrode charge transfer time scale [s]": 500,
            "Positive electrode capacitance [F]": 1,
            "Negative electrode capacitance [F]": 1,
            "Cation transference number": 0.25,
            "Positive electrode relative thickness": 0.47,
            "Negative electrode relative thickness": 0.47,
            "Series resistance [Ohm]": 0.01,
        }
        return ParameterValues(values=parameter_dictionary)

    @property
    def default_quick_plot_variables(self):
        return [
            "Negative particle surface stoichiometry",
            "Electrolyte stoichiometry",
            "Positive particle surface stoichiometry",
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


def convert_physical_to_grouped_parameters(parameter_set):
    """
    A function to create a grouped SPMe parameter set from a standard
    PyBaMM parameter set.

    Parameters
    ----------
    parameter_set : Union[dict, pybop.ParameterSet, pybamm.ParameterValues]
        A dict-like object containing the parameter values.

    Returns
    -------
    dict
        A dictionary of the grouped parameters.
    """
    parameter_set = ParameterSet.to_pybamm(parameter_set)

    # Unpack physical parameters
    F = parameter_set["Faraday constant [C.mol-1]"]
    alpha_p = parameter_set["Positive electrode active material volume fraction"]
    alpha_n = parameter_set["Negative electrode active material volume fraction"]
    c_max_p = parameter_set["Maximum concentration in positive electrode [mol.m-3]"]
    c_max_n = parameter_set["Maximum concentration in negative electrode [mol.m-3]"]
    L_p = parameter_set["Positive electrode thickness [m]"]
    L_n = parameter_set["Negative electrode thickness [m]"]
    epsilon_p = parameter_set["Positive electrode porosity"]
    epsilon_n = parameter_set["Negative electrode porosity"]
    R_p = parameter_set["Positive particle radius [m]"]
    R_n = parameter_set["Negative particle radius [m]"]
    D_p = parameter_set["Positive particle diffusivity [m2.s-1]"]
    D_n = parameter_set["Negative particle diffusivity [m2.s-1]"]
    b_p = parameter_set["Positive electrode Bruggeman coefficient (electrolyte)"]
    b_n = parameter_set["Negative electrode Bruggeman coefficient (electrolyte)"]
    Cdl_p = parameter_set["Positive electrode double-layer capacity [F.m-2]"]
    Cdl_n = parameter_set["Negative electrode double-layer capacity [F.m-2]"]
    m_p = 3.42e-6  # (A/m2)(m3/mol)**1.5
    m_n = 6.48e-7  # (A/m2)(m3/mol)**1.5
    sigma_p = parameter_set["Positive electrode conductivity [S.m-1]"]
    sigma_n = parameter_set["Negative electrode conductivity [S.m-1]"]

    # Separator and electrolyte properties
    ce0 = parameter_set["Initial concentration in electrolyte [mol.m-3]"]
    De = parameter_set["Electrolyte diffusivity [m2.s-1]"]  # (ce0, T)
    L_s = parameter_set["Separator thickness [m]"]
    epsilon_sep = parameter_set["Separator porosity"]
    b_sep = parameter_set["Separator Bruggeman coefficient (electrolyte)"]
    t_plus = parameter_set["Cation transference number"]
    sigma_e = parameter_set["Electrolyte conductivity [S.m-1]"]  # (ce0, T)

    # Compute the cell area and thickness
    A = parameter_set["Electrode height [m]"] * parameter_set["Electrode width [m]"]
    L = L_p + L_n + L_s

    # Compute the series resistance
    Re = (
        L_p / (3 * epsilon_p**b_p)
        + L_s / (epsilon_sep**b_sep)
        + L_n / (3 * epsilon_n**b_n)
    ) / sigma_e
    Rs = (L_p / sigma_p + L_n / sigma_n) / 3
    R0 = Re + Rs + parameter_set["Contact resistance [Ohm]"]

    # Compute the stoichiometry limits and initial SOC
    x_0, x_100, y_100, y_0 = get_min_max_stoichiometries(parameter_set)
    sto_p_init = (
        parameter_set["Initial concentration in positive electrode [mol.m-3]"] / c_max_p
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
    Q_e = F * epsilon_sep * ce0 * L * A

    zeta_p = epsilon_p / epsilon_sep
    zeta_n = epsilon_n / epsilon_sep

    tau_d_p = R_p**2 / D_p
    tau_d_n = R_n**2 / D_n

    tau_e_p = epsilon_sep * L**2 / (epsilon_p**b_p * De)
    tau_e_n = epsilon_sep * L**2 / (epsilon_n**b_n * De)
    tau_e_sep = epsilon_sep * L**2 / (epsilon_sep**b_sep * De)

    tau_ct_p = F * R_p / (m_p * np.sqrt(ce0))
    tau_ct_n = F * R_n / (m_n * np.sqrt(ce0))

    C_p = 3 * alpha_p * Cdl_p * L_p * A / R_p
    C_n = 3 * alpha_n * Cdl_n * L_n * A / R_n

    l_p = L_p / L
    l_n = L_n / L

    return {
        "Nominal cell capacity [A.h]": parameter_set["Nominal cell capacity [A.h]"],
        "Current function [A]": parameter_set["Current function [A]"],
        "Initial temperature [K]": parameter_set["Ambient temperature [K]"],
        "Initial SoC": soc_init,
        "Minimum negative stoichiometry": x_0,
        "Maximum negative stoichiometry": x_100,
        "Minimum positive stoichiometry": y_100,
        "Maximum positive stoichiometry": y_0,
        "Lower voltage cut-off [V]": parameter_set["Lower voltage cut-off [V]"],
        "Upper voltage cut-off [V]": parameter_set["Upper voltage cut-off [V]"],
        "Positive electrode OCP [V]": parameter_set["Positive electrode OCP [V]"],
        "Negative electrode OCP [V]": parameter_set["Negative electrode OCP [V]"],
        "Measured cell capacity [A.s]": Q_meas,
        "Reference electrolyte capacity [A.s]": Q_e,
        "Positive electrode relative porosity": zeta_p,
        "Negative electrode relative porosity": zeta_n,
        "Positive particle diffusion time scale [s]": tau_d_p,
        "Negative particle diffusion time scale [s]": tau_d_n,
        "Positive electrode electrolyte diffusion time scale [s]": tau_e_p,
        "Negative electrode electrolyte diffusion time scale [s]": tau_e_n,
        "Separator electrolyte diffusion time scale [s]": tau_e_sep,
        "Positive electrode charge transfer time scale [s]": tau_ct_p,
        "Negative electrode charge transfer time scale [s]": tau_ct_n,
        "Positive electrode capacitance [F]": C_p,
        "Negative electrode capacitance [F]": C_n,
        "Cation transference number": t_plus,
        "Positive electrode relative thickness": l_p,
        "Negative electrode relative thickness": l_n,
        "Series resistance [Ohm]": R0,
    }
