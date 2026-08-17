import pybamm

""" Alternative functions written as classes to allow pickling. """


class FunctionalDiffusionTime:
    def __init__(self, r2_scale, D, c_scale):
        self.r2_scale = r2_scale
        self.D = D
        self.c_scale = c_scale

    def __call__(self, sto, T):
        r2_scale, D, c_scale = self.r2_scale, self.D, self.c_scale
        return r2_scale / D(sto * c_scale, T)


class AsymmetricButlerVolmer:
    def __init__(self, alpha):
        self.alpha = alpha  # cathodic transfer coefficient

    def __call__(self, sto_surf, sto_e, eta_RT_F):
        alpha = self.alpha
        j0 = sto_surf**alpha * (sto_e * (1 - sto_surf)) ** (1 - alpha)
        return j0 * (pybamm.exp((1 - alpha) * eta_RT_F) - pybamm.exp(-alpha * eta_RT_F))


class MultiphaseButlerVolmer:
    def __init__(self, alpha, omega):
        self.alpha = alpha
        self.omega = omega

    def __call__(self, sto_surf, sto_e, eta_RT_F):
        alpha, omega = self.alpha, self.omega
        j0 = (
            sto_surf ** (alpha * omega)
            * (1 - sto_surf) ** ((1 - alpha) * omega)
            * sto_e ** (1 - alpha)
        )
        return j0 * (pybamm.exp((1 - alpha) * eta_RT_F) - pybamm.exp(-alpha * eta_RT_F))
