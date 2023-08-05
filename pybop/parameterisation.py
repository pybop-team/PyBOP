import pybop
import pybamm
import numpy as np


# To Do:
# checks on observations and parameters
# implement method to update parameterisation without reconstructing the simulation
# Geometric parameters might always require reconstruction (i.e. electrode height)


class Parameterisation:
    """
    Parameterisation class for pybop.
    """

    def __init__(self, model, observations, fit_parameters, x0=None, options=None):
        self.model = model
        self.fit_parameters = fit_parameters
        self.observations = observations
        self.options = options
        self.bounds = dict(
            lower=[self.fit_parameters[p].bounds[0] for p in self.fit_parameters],
            upper=[self.fit_parameters[p].bounds[1] for p in self.fit_parameters],
        )

        if model.pybamm_model is not None:
            if model.parameter_set is not None:
                self.parameter_set = model.parameter_set
            else:
                self.parameter_set = self.model.pybamm_model.default_parameter_values

            try:
                self.parameter_set["Current function [A]"] = pybamm.Interpolant(
                    self.observations["Time [s]"].data,
                    self.observations["Current function [A]"].data,
                    pybamm.t,
                )
            except:
                raise ValueError("Current function not supplied")

            if x0 is None:
                self.x0 = np.mean([self.bounds["lower"], self.bounds["upper"]], axis=0)

            # self.sim = pybop.Simulation(
            #     self.model.pybamm_model,
            #     parameter_values=self.parameter_set,
            #     solver=pybamm.CasadiSolver(),
            # )

            # set input parameters in parameter set from fitting parameters
            for i in self.fit_parameters:
                self.parameter_set[i] = "[input]"
            
            self.fit_dict = {p: self.fit_parameters[p].prior.mean for p in self.fit_parameters}

            #Set up geometry on model
            geometry = self.model.pybamm_model.default_geometry

            # Set up parameters for geometry and model
            self.parameter_set.process_model(self.model.pybamm_model)
            self.parameter_set.process_geometry(geometry)

            # Mesh the model
            mesh = pybamm.Mesh(geometry, self.model.pybamm_model.default_submesh_types, self.model.pybamm_model.default_var_pts) #update

            # Discretise the model
            disc = pybamm.Discretisation(mesh, self.model.pybamm_model.default_spatial_methods)
            disc.process_model(self.model.pybamm_model)

            # Set solver
            self.solver = pybamm.CasadiSolver()
        else:
            raise ValueError("No pybamm model supplied")


    def map(self, x0):
        """
        Max a posteriori estimation.
        """
        pass

    def sample(self, n_chains):
        """
        Sample from the posterior distribution.
        """
        pass

    def pem(self, method=None):
        """
        Predictive error minimisation.
        """
        pass

    def rmse(self, method=None):
        """
        Calculate the root mean squared error.
        """

        def step(x, grad):
            for i,p in enumerate(self.fit_dict):
                self.fit_dict[p] = x[i]
            print(self.fit_dict)

            # self.sim = pybamm.Simulation(
            #     self.model.pybamm_model,
            #     parameter_values=self.parameter_set,
            #     solver=pybamm.CasadiSolver(),
            # )
            y_hat = self.solver.solve(self.model.pybamm_model, inputs=self.fit_dict)["Terminal voltage [V]"].data

            try:
                res = np.sqrt(
                    np.mean((self.observations["Voltage"].data[1] - y_hat) ** 2)
                )
            except:
                raise ValueError(
                    "Measurement and modelled data length mismatch, potentially due to voltage cut-offs"
                )
            return res

        if method == "nlopt":
            optim = pybop.NLoptOptimize(x0=self.x0)
            results = optim.optimise(step, self.x0, self.bounds)
        else:
            optim = pybop.NLoptOptimize(method)
            results = optim.optimise(step, self.x0, self.bounds)
        return results

    def mle(self, method=None):
        """
        Maximum likelihood estimation.
        """
        pass