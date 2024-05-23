#
# Weppner Huggins Model
#
import numpy as np
from pybamm import DummySolver, Parameter, ParameterValues, citations
from pybamm import lithium_ion as pybamm_lithium_ion
from pybamm import t as pybamm_t


class BaseWeppnerHuggins(pybamm_lithium_ion.BaseModel):
    """WeppnerHuggins Model for GITT. Credit: pybamm-param team.

    Parameters
    ----------
    name : str, optional
        The name of the model.
    """

    def __init__(self, name="Weppner & Huggins model"):
        super().__init__({}, name)

        citations.register("""
            @article{Weppner1977,
            title={{Determination of the kinetic parameters
            of mixed-conducting electrodes and application to the system Li3Sb}},
            author={Weppner, W and Huggins, R A},
            journal={Journal of The Electrochemical Society},
            volume={124},
            number={10},
            pages={1569},
            year={1977},
            publisher={IOP Publishing}
            }
        """)

        # `self.param` is a class containing all the relevant parameters and functions for
        # this model. These are purely symbolic at this stage, and will be set by the
        # `ParameterValues` class when the model is processed.
        self.options["working electrode"] = "positive"
        self._summary_variables = []

        t = pybamm_t
        ######################
        # Parameters
        ######################

        d_s = Parameter("Positive electrode diffusivity [m2.s-1]")

        c_s_max = Parameter("Maximum concentration in positive electrode [mol.m-3]")

        i_app = self.param.current_density_with_time

        U = Parameter("Reference OCP [V]")

        U_prime = Parameter("Derivative of the OCP wrt stoichiometry [V]")

        epsilon = Parameter("Positive electrode active material volume fraction")

        r_particle = Parameter("Positive particle radius [m]")

        a = 3 * (epsilon / r_particle)

        l_w = self.param.p.L

        ######################
        # Governing equations
        ######################
        u_surf = (
            (2 / (np.pi**0.5))
            * (i_app / ((d_s**0.5) * a * self.param.F * l_w))
            * (t**0.5)
        )
        # Linearised voltage
        V = U + (U_prime * u_surf) / c_s_max
        ######################
        # (Some) variables
        ######################
        self.variables = {
            "Voltage [V]": V,
            "Time [s]": t,
        }

    @property
    def default_geometry(self):
        return {}

    @property
    def default_parameter_values(self):
        parameter_values = ParameterValues("Xu2019")
        parameter_values.update(
            {
                "Reference OCP [V]": 4.1821,
                "Derivative of the OCP wrt stoichiometry [V]": -1.38636,
            },
            check_already_exists=False,
        )
        return parameter_values

    @property
    def default_submesh_types(self):
        return {}

    @property
    def default_var_pts(self):
        return {}

    @property
    def default_spatial_methods(self):
        return {}

    @property
    def default_solver(self):
        return DummySolver()
