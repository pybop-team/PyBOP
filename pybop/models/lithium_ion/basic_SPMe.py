import warnings

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
from pybamm.input.parameters.lithium_ion.Chen2020 import (
    graphite_LGM50_ocp_Chen2020,
    nmc_LGM50_ocp_Chen2020,
)


class BaseGroupedSPMe(pybamm_lithium_ion.BaseModel):
    """
    A grouped parameter version of the single particle model with electrolyte (SPMe).

    Parameters
    ----------
    name : str, optional
        The name of the model.
    """

    def __init__(self, name="Single particle model with electrolyte", **model_kwargs):
        if model_kwargs is not dict(build=True):
            unused_kwargs_warning = f"The input model_kwargs {list(model_kwargs.keys())} are not currently used by the SPMe."
            warnings.warn(unused_kwargs_warning, UserWarning, stacklevel=2)

        super().__init__({}, name=name)

        pybamm.citations.register("Chen2020")  # for the OCPs
        pybamm.citations.register("""
            @article{HallemansPreprint,
            title={{Hallemans Preprint}},
            }
        """)

        ######################
        # Variables
        ######################
        # Variables that depend on time only are created without a domain
        Q = Variable("Discharge capacity [A.h]")
        v_n = Variable("Negative electrode voltage [V]")
        v_p = Variable("Positive electrode voltage [V]")
        j_n = Variable("Negative molar flux")
        j_p = Variable("Positive molar flux")

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
            "Negative electrolyte stoichiometry",
            domain="negative electrode",
        )
        sto_e_s = Variable(
            "Separator electrolyte stoichiometry",
            domain="separator",
        )
        sto_e_p = Variable(
            "Positive electrolyte stoichiometry",
            domain="positive electrode",
        )
        # Concatenations combine several variables into a single variable, to simplify
        # implementing equations that hold over several domains
        sto_e = pybamm.concatenation(sto_e_n, sto_e_s, sto_e_p)

        # Surf takes the surface value of a variable, i.e. its boundary value on the
        # right side. This is also accessible via `boundary_value(x, "right")`, with
        # "left" providing the boundary value of the left side
        sto_n_surf = pybamm.surf(sto_n)
        sto_p_surf = pybamm.surf(sto_p)

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
        Q_th_p = Parameter("Positive theoretical electrode capacity [As]")
        Q_th_n = Parameter("Negative theoretical electrode capacity [As]")

        tau_d_p = Parameter("Positive particle diffusion time scale [s]")
        tau_d_n = Parameter("Negative particle diffusion time scale [s]")

        tau_r_p = Parameter("Positive electrode charge transfer time scale [s]")
        tau_r_n = Parameter("Negative electrode charge transfer time scale [s]")

        C_p = Parameter("Positive electrode capacitance [F]")
        C_n = Parameter("Negative electrode capacitance [F]")

        t_plus = Parameter("Cation transference number")

        l_p = Parameter("Positive electrode relative thickness")
        l_n = Parameter("Negative electrode relative thickness")

        R0 = Parameter("Series resistance [Ohm]")

        # Primary broadcasts are used to broadcast scalar quantities across a domain
        # into a vector of the right shape, for multiplying with other vectors
        tau_e_n = PrimaryBroadcast(
            Parameter("Negative electrode electrolyte diffusion time scale [s]"),
            "negative electrode",
        )
        tau_e_sep = PrimaryBroadcast(
            Parameter("Separator electrolyte diffusion time scale [s]"),
            "separator",
        )
        tau_e_p = PrimaryBroadcast(
            Parameter("Positive electrode electrolyte diffusion time scale [s]"),
            "positive electrode",
        )
        tau_e = pybamm.concatenation(tau_e_n, tau_e_sep, tau_e_p)

        beta_n = PrimaryBroadcast(
            Parameter("Negative relative concentration"),
            "negative electrode",
        )
        beta_sep = PrimaryBroadcast(
            Scalar(0),
            "separator",
        )
        beta_p = PrimaryBroadcast(
            Parameter("Positive relative concentration"),
            "positive electrode",
        )

        ######################
        # Input current
        ######################
        I = self.param.current_with_time

        ######################
        # State of Charge
        ######################
        # The `rhs` dictionary contains differential equations, with the key being the
        # variable in the d/dt
        self.rhs[Q] = I / 3600
        # Initial conditions must be provided for the ODEs
        self.initial_conditions[Q] = Scalar(0)

        ######################
        # Overpotentials
        ######################
        j0_n = pybamm.sqrt(sto_n_surf * (1 - sto_n_surf)) * (
            pybamm.x_average(pybamm.sqrt(sto_e_n))
        )
        j0_p = pybamm.sqrt(sto_p_surf * (1 - sto_p_surf)) * (
            pybamm.x_average(pybamm.sqrt(sto_e_p))
        )

        eta_n = 2 * RT_F * pybamm.arcsinh(tau_r_n * j_n / (2 * j0_n))
        eta_p = 2 * RT_F * pybamm.arcsinh(tau_r_p * j_p / (2 * j0_p))
        eta_e = (2 * RT_F * (1 - t_plus)) * pybamm.log(
            pybamm.x_average(sto_e_p) / pybamm.x_average(sto_e_n)
        )

        # The `algebraic` dictionary contains differential equations, with the key being
        # the main scalar variable of interest in the equation
        self.algebraic[j_n] = self.U(sto_n_surf, "negative") + eta_n - v_n
        self.algebraic[j_p] = self.U(sto_p_surf, "positive") + eta_p - v_p

        # Initial conditions must also be provided for algebraic equations, as an
        # initial guess for a root-finding algorithm which calculates consistent
        # initial conditions
        j_n_init = I / (3 * Q_th_n)  # estimating dv_n/dt=0
        j_p_init = -I / (3 * Q_th_p)  # estimating dv_p/dt=0
        self.initial_conditions[j_n] = j_n_init
        self.initial_conditions[j_p] = j_p_init

        ######################
        # Double-layer
        ######################
        self.rhs[v_n] = 1 / C_n * (-Q_th_n * j_n + I / 3)
        self.rhs[v_p] = 1 / C_p * (-Q_th_p * j_p - I / 3)

        sto_n_init = x_0 + (x_100 - x_0) * soc_init
        sto_p_init = y_0 + (y_100 - y_0) * soc_init
        U_n_init = self.U(sto_n_init, "negative")
        U_p_init = self.U(sto_p_init, "positive")
        eta_n_init = (2 * RT_F) * pybamm.arcsinh(
            (tau_r_n * j_n_init) / (2 * pybamm.sqrt(sto_n_init * (1 - sto_n_init)))
        )
        eta_p_init = -(2 * RT_F) * pybamm.arcsinh(
            (tau_r_p * j_p_init) / (2 * pybamm.sqrt(sto_p_init * (1 - sto_p_init)))
        )
        self.initial_conditions[v_n] = U_n_init + eta_n_init
        self.initial_conditions[v_p] = U_p_init + eta_p_init

        ######################
        # Particles
        ######################
        # The div and grad operators will be converted to the appropriate matrix
        # multiplication at the discretisation stage
        self.rhs[sto_n] = pybamm.div(1 / tau_d_n * pybamm.grad(sto_n))
        self.rhs[sto_p] = pybamm.div(1 / tau_d_p * pybamm.grad(sto_p))

        # Boundary conditions must be provided for equations with spatial derivatives
        self.boundary_conditions[sto_n] = {
            "left": (Scalar(0), "Neumann"),
            "right": (-tau_d_n * j_n, "Neumann"),
        }
        self.boundary_conditions[sto_p] = {
            "left": (Scalar(0), "Neumann"),
            "right": (-tau_d_p * j_p, "Neumann"),
        }

        self.initial_conditions[sto_n] = sto_n_init
        self.initial_conditions[sto_p] = sto_p_init

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
        # Electrolyte
        ######################
        b_e_n = 3 * beta_n * (1 - t_plus) * j_n
        b_e_sep = beta_sep
        b_e_p = 3 * beta_p * (1 - t_plus) * j_p
        beta = pybamm.concatenation(b_e_n, b_e_sep, b_e_p)
        self.rhs[sto_e] = pybamm.div(1 / tau_e * pybamm.grad(sto_e)) + beta

        self.boundary_conditions[sto_e] = {
            "left": (Scalar(0), "Neumann"),
            "right": (Scalar(0), "Neumann"),
        }

        self.initial_conditions[sto_e] = Scalar(1)

        ######################
        # Cell voltage
        ######################
        V = v_p - v_n + eta_e - R0 * I

        # Save the initial OCV
        self.ocv_init = U_p_init - U_n_init

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
        # Primary broadcasts are used to broadcast scalar quantities across a domain
        # into a vector of the right shape, for multiplying with other vectors
        self.variables = {
            "Negative particle stoichiometry": sto_n,
            "Negative particle surface stoichiomtry": PrimaryBroadcast(
                sto_n_surf, "negative electrode"
            ),
            # "Negative electrode voltage [V]": PrimaryBroadcast(
            #     v_n, "negative electrode"
            # ),
            "Negative electrolyte stoichiometry": sto_e_n,
            "Separator electrolyte stoichiometry": sto_e_s,
            "Positive electrolyte stoichiometry": sto_e_p,
            "Positive particle stoichiometry": sto_p,
            "Positive particle surface stoichiometry": PrimaryBroadcast(
                sto_p_surf, "positive electrode"
            ),
            # "Positive electrode voltage [V]": PrimaryBroadcast(
            #     v_p, "positive electrode"
            # ),
            "Time [s]": pybamm_t,
            "Current [A]": I,
            "Current variable [A]": I,  # for compatibility with pybamm.Experiment
            "Discharge capacity [A.h]": Q,
            "Voltage [V]": V,
        }

    @property
    def default_parameter_values(self):
        parameter_dictionary = {
            "Current function [A]": 0,
            "Nominal cell capacity [A.h]": 3,
            "Initial temperature [K]": 298,
            "Initial SoC": 0.5,
            "Minimum negative stoichiometry": 0.026,
            "Maximum negative stoichiometry": 0.911,
            "Minimum positive stoichiometry": 0.264,
            "Maximum positive stoichiometry": 0.854,
            "Positive electrode thickness [m]": 0.438,  # normalised
            "Negative electrode thickness [m]": 0.493,  # normalised
            "Separator thickness [m]": 0.069,  # normalised
            "Positive particle radius [m]": 1,  # normalised
            "Negative particle radius [m]": 1,  # normalised
            "Lower voltage cut-off [V]": 2.5,
            "Upper voltage cut-off [V]": 4.2,
            "Positive electrode OCP [V]": nmc_LGM50_ocp_Chen2020,
            "Negative electrode OCP [V]": graphite_LGM50_ocp_Chen2020,
            "Positive theoretical electrode capacity [As]": 3000,
            "Negative theoretical electrode capacity [As]": 3000,
            "Positive relative concentration": 1,
            "Negative relative concentration": 1,
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

    def build_model(self):
        # Build model variables and equations
        self._build_model()

        # # Set battery specific variables
        # pybamm.logger.debug(f"Setting voltage variables ({self.name})")
        # self.set_voltage_variables()

        # pybamm.logger.debug(f"Setting SoC variables ({self.name})")
        # self.set_soc_variables()

        # pybamm.logger.debug(f"Setting degradation variables ({self.name})")
        # self.set_degradation_variables()
        # self.set_summary_variables()

        self._built = True
        pybamm.logger.info(f"Finish building {self.name}")

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
        # see #1435
        out = u_ref + 1e-6 * (1 / sto + 1 / (sto - 1))

        if domain == "negative":
            out.print_name = r"U_\mathrm{n}(c^\mathrm{surf}_\mathrm{s,n})"
        elif domain == "positive":
            out.print_name = r"U_\mathrm{p}(c^\mathrm{surf}_\mathrm{s,p})"
        return out
