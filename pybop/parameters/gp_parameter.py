import itertools
from typing import Any

import numpy as np
import pybamm

from pybop.parameters.distributions import Gaussian
from pybop.parameters.parameter import Parameter
from pybop.pybamm.parameter_utils import ParameterValues

try:
    from FoKL.getKernels import bernoulli, sp500

    FOKL_AVAILABLE = True
except ImportError:
    FOKL_AVAILABLE = False


class FoKLGP:
    """
    Creates parameter functions as decomposed GPs
    """

    def __init__(
        self,
        name,
        options=None,
        parameter_values=None,
        kernel="Bernoulli Polynomial",
        twoway=True,
        model=None,
    ):
        """
        Parameters
        ----------
        name : str
            Name of the parameter being processed
        options : dict
            options dictionary for GP creation
        kernel : str
            sets GP kernel, only "Bernoulli Polynomials

        """
        if not FOKL_AVAILABLE:
            raise ModuleNotFoundError(
                "The `FoKL` package is required to use FoKLGP objects. "
                "Please install it using: pip install FoKL"
            )
        self.model = model
        GP_dict_list = self._process_options(name, options, parameter_values)
        self.evaluate_func = self._evaluate_parameter(kernel)
        self.damtx = self._create_interaction_matrix(
            GP_dict_list["Number of terms"], GP_dict_list["Number of inputs"], twoway
        )
        self.add_to_params(GP_dict_list, self.damtx)

    def __call__(self, *args):
        return self.pybamm_function(*args)

    def _process_options(
        self, name: str, options: dict[Any], parameter_values: ParameterValues
    ):
        """

        Process configuration options for a FoKLGP object and apply defaults.

        Parameters
        ----------
        name : str
            Name of the parameter being processed.
        options : dict or None
            User-supplied configuration options to override defaults. Expected
            dictionary keys include:
                * 'arg_inds' (list[int]): Indices of PyBAMM function arguments.
                * 'div_arg' (list[list[int]]): Indices for division terms, e.g.,
                  [[3, 2]] computes input 3 divided by input 2.
                * 'div_const' (int): Normalization term to scale inputs between 0 and 1.
                * 'inv_arg' (list[int]): Indices to invert, e.g., [3] returns 1 / input 3.
                * 'Number of terms' (int): Model order depth (e.g., 3 creates 7 terms).
                * 'Constant mean' (float): Mean for the first Beta parameter (B0).
                  Inferred from `parameter_values` if not provided.
                * 'Constant standard deviation' (float): Standard deviation for B0.
                * 'Bi mean' (float): Mean for subsequent Beta parameters (Bi).
                * 'Bi standard deviation' (float): Standard deviation for Bi.
                * 'exp' (bool): If True, applies log-transformation to B0_mean.
                * 'Normalization min-max' (dict[str(int):tuple]) : Normalization minimum and maximum for `arg_ind` terms
                                                                  (e.g, {'0':(10,20)} results in argument 0 being
                                                                  normalized between 10 - 20.
                * 'Verbose' (bool): Show debugging print statements
        parameter_values : dict or Mapping
            Dictionary containing base parameter values, used to calculate
            'Constant mean' if it is missing from options.

        Returns
        -------
        dict
            The finalized options dictionary containing both defaults and
            user-defined overrides.

        Raises
        ------
        ValueError
            If 'Constant mean' is missing and the value in `parameter_values`
            cannot be resolved to a constant.

        """
        default_options = {
            "Arguments": None,
            "exp": True,
            "Number of inputs": 1,
            "Division Arguments": None,
            "Number of terms": 1,
            "Constant standard deviation": 0.2,
            "Bi mean": 0,
            "Bi standard deviation": 0.2,
            "Normalization min-max": {},
            "Verbose": False,
        }
        default_options.update({"Name": name})
        if options is not None:
            default_options.update(options)
        if "Constant mean" not in default_options:
            # If no beta constant term distribution described grab from parameters,
            # if this is a function then user supplied
            try:
                if default_options["exp"]:
                    B0_mean = np.log(parameter_values[name])

                else:
                    B0_mean = parameter_values[name]
                default_options.update({"Constant mean": B0_mean})
            except (TypeError, KeyError) as err:
                raise ValueError(
                    f"Default parameter value for {name} is not a constant, please supply an estimate"
                ) from err

        num_inputs = 0

        default_options["Input names"] = self._check_arguments(default_options)

        if default_options["Arguments"] is not None:
            num_inputs += len(default_options["Arguments"])
        if default_options["Division Arguments"] is not None:
            num_inputs += len(default_options["Division Arguments"])

        for arg in default_options["Arguments"] or []:
            if arg not in default_options["Normalization min-max"]:
                default_options["Normalization min-max"] = {arg: (0, 1)}
        self.verbose = default_options["Verbose"]
        default_options["Number of inputs"] = num_inputs
        self.parameter_values = parameter_values
        return default_options

    def _check_arguments(self, GP_options):
        input_names = self._get_function_parameter_input_names(
            self.model, GP_options["Name"]
        )
        if GP_options["Arguments"] is not None:
            for n in GP_options["Arguments"]:
                if n not in input_names:
                    raise ValueError(
                        f"Input argument {n} not found. Possible inputs are {input_names}"
                    )
        if GP_options["Division Arguments"] is not None:
            for div_arg in GP_options["Division Arguments"]:
                for n in div_arg:
                    if n not in input_names:
                        raise ValueError(
                            f"Input argument {n} not found. Possible inputs are {input_names}"
                        )
        return input_names

    def _create_interaction_matrix(
        self, number_of_terms, number_of_inputs, twoway, damtx=None
    ):
        """
        Creates interaction matrix, defines terms of model expansion
        """

        def perms(x):
            """Python equivalent of MATLAB perms."""
            a = np.array(np.vstack(list(itertools.permutations(x)))[::-1])
            return a

        def sum_to_n(n, size, limit=None):
            """Produce all lists of `size` positive integers in decreasing order
            that add up to `n`."""
            if size == 1:
                yield [n]
                return
            if limit is None:
                limit = n
            start = (n + size - 1) // size
            stop = min(limit, n - size + 1) + 1
            for i in range(start, stop):
                for tail in sum_to_n(n - i, size - 1, i):
                    yield [i] + tail

        if twoway:
            sett = 2
        else:
            sett = 1
        if number_of_inputs == 1:
            damtx = (
                np.linspace(1, number_of_terms, number_of_terms)
                .astype(int)
                .reshape(-1, 1)
            )
        else:
            principle = np.zeros((number_of_inputs,))
            if damtx is None:
                damtx = []
            for ind in range(1, number_of_terms + 1):
                indvecs = [i for i in sum_to_n(ind, size=min(number_of_inputs, sett))]
                principle[0] = ind
                indvecs.append(list(principle))
                for indvec in indvecs:
                    new_term = perms(indvec)
                    if np.size(damtx) == 0:
                        damtx = new_term
                    else:
                        if all(new_term[0] == new_term[1]):
                            damtx = np.vstack([damtx, new_term[0]])
                        else:
                            damtx = np.vstack([damtx, new_term])

                    indvec[0] += 1
        damtx = np.array(damtx)
        if self.verbose:
            print(damtx)
        return damtx.astype(int)

    def _set_kernel(self, kernel="Bernoulli Polynomial"):
        if kernel == "Cubic Splines":
            self.phis = sp500()
        elif kernel == "Bernoulli Polynomial":
            self.phis = bernoulli()

    def _evaluate_parameter(self, kernel):
        """
        The symbolic evaluation of the decomposed GP.

        kernel: Kernel function for evaluation, must be `Cubic Splines` or `Bernoulli Polynomial`

        Returns:
        ---------
        evaluate_pybamm : function
            Symbolic GP function for defined kernel
        """
        self._set_kernel(kernel)

        if kernel == "Cubic Splines":

            def evaluate_pybamm(betas, mtx, inputs, coeff=None):

                num_basis_terms = len(mtx)
                num_inputs = len(mtx[0])
                X_sol = []

                mtx = np.array(mtx)
                phind = []
                X_sc = []
                A = [1, 2, 3]
                for i in range(num_inputs):
                    phind_temp = inputs[i][0] * 499
                    sett = pybamm.EqualHeaviside(0, phind_temp)
                    phind_temp = phind_temp + sett
                    phind.append(phind_temp - 1)
                    X_sc.append([(1 - inputs[0][i]) ** a for a in A])

                lspace = np.linspace(0, 499, 499)

                for j in range(num_basis_terms):
                    phi = 1
                    for k in range(num_inputs):
                        num = mtx[j][k]

                        if num > 0:
                            nid = int(num - 1)
                            if coeff is None:
                                coeff = []
                                for jj in range(4):
                                    phispace = self.phis[nid][jj].reshape(1, -1)
                                    phi_interp = pybamm.Interpolant(
                                        lspace, phispace[0], phind[k]
                                    )
                                    coeff.append(phi_interp)

                            phi *= (
                                coeff[0]
                                + coeff[1] * X_sc[k][0]
                                + coeff[2] * X_sc[k][1]
                                + coeff[3] * X_sc[k][2]
                            )
                        coeff = None
                        X_sol.append(phi)

                X_sol_ones = betas[0]
                mean = X_sol_ones
                for i in range(len(X_sol)):
                    X_sol_betas = X_sol[i] * betas[i + 1]
                    mean += X_sol_betas

                return mean
        elif kernel == "Bernoulli Polynomial":

            def evaluate_pybamm(betas, mtx, inputs, coeff=None):

                num_basis_terms = len(mtx)
                num_inputs = len(mtx[0])
                X_sol = []

                mtx = np.array(mtx)

                def bernoulli_func(phis, num, x):
                    if num > 0:
                        coeff = phis[num - 1]
                        result = coeff[0] + sum(
                            coeff[k] * (x**k) for k in range(1, len(coeff))
                        )
                    else:
                        result = 1.0
                    return result

                for j in range(num_basis_terms):
                    phi = 1
                    for k in range(num_inputs):
                        num = mtx[j][k]
                        phi *= bernoulli_func(self.phis, num, inputs[k][0])
                    X_sol.append(phi)

                X_sol_ones = betas[0]
                mean = X_sol_ones
                for i in range(len(X_sol)):
                    X_sol_betas = X_sol[i] * betas[i + 1]
                    mean += X_sol_betas
                return mean
        else:
            raise NotImplementedError(
                "Kernel must be either `Cubic Splines` or `Bernoulli Polynomial`"
            )

        return evaluate_pybamm

    @staticmethod
    def _get_function_parameter_input_names(model, name):
        """
        Return the ordered, PyBaMM-standard descriptive input names for the
        FunctionParameter called `name`, e.g.
        ["Electrolyte concentration [mol.m-3]", "Temperature [K]"].

        `model` must be the un-built/un-discretised pybamm.BaseModel — once
        parameters are processed, FunctionParameter nodes are replaced by
        plain Function nodes and this info is gone.
        """
        info = model.get_parameter_info()
        for var_symbol, _ in info.values():
            if (
                isinstance(var_symbol, pybamm.FunctionParameter)
                and var_symbol.name == name
            ):
                return list(var_symbol.input_names)
        raise ValueError(
            f"No FunctionParameter named '{name}' found in the model. "
            "Call model.print_parameter_info() to see the available names."
        )

    @staticmethod
    def _unpack_str_inputs(arguments, input_names):
        arg_inds = []
        for arg in arguments:
            pos = input_names.index(arg)
            arg_inds.append(pos)
        return arg_inds

    @staticmethod
    def _unpack_div_str_inputs(div_args_str, input_names):
        div_arg = []
        for term in div_args_str:
            num = term[0]
            dom = term[1]
            div_arg.append([num, dom])
        return div_arg

    def add_function(
        self,
        name,
        mtx,
        betas_function,
        input_names,
        arguments=None,
        division_arguments=None,
        norm_bounds=None,
        exp=False,
    ):
        """
        Creates Parameter function specified as a GP object
        """

        beta_func = betas_function

        if arguments is None:
            arg_inds = []
        else:
            arg_inds = self._unpack_str_inputs(arguments, input_names)
        if division_arguments is None:
            div_arg = []
        else:
            div_arg = self._unpack_div_str_inputs(division_arguments, input_names)

        if exp:

            def pybamm_function(*args):
                xs = []
                for x in div_arg:
                    xs.append([args[x[0]] / args[x[1]]])

                for x in arg_inds:
                    xs.append(
                        [
                            (args[x] - norm_bounds[str(x)][0])
                            / (norm_bounds[str(x)][1] - norm_bounds[str(x)][0])
                        ]
                    )

                res = np.exp(self.evaluate_func(beta_func, mtx, xs))
                return res
        else:

            def pybamm_function(*args):
                xs = []
                for x in div_arg:
                    xs.append([args[x[0]] / args[x[1]]])
                for x in arg_inds:
                    xs.append(
                        [
                            (args[x] - norm_bounds[input_names[x]][0])
                            / (
                                norm_bounds[input_names[x]][1]
                                - norm_bounds[input_names[x]][0]
                            )
                        ]
                    )
                res = self.evaluate_func(beta_func, mtx, xs)
                return res

        if self.verbose:
            function_args_mod = []
            if division_arguments is not None:
                for x in division_arguments:
                    function_args_mod.append(x[0] + "/" + x[1])
            if arguments is not None:
                for x in arguments:
                    function_args_mod.append(
                        "("
                        + x
                        + str(f" - {norm_bounds[x][0]}) / ")
                        + str(f"({norm_bounds[x][1] - norm_bounds[x][0]})")
                    )
            print(f"GP function created for {name} \n inputs are {function_args_mod}")
        self.pybamm_function = pybamm_function
        return pybamm_function

    def create_beta_inputs(self, len_mtx, GP):
        """
        Generates Input Variables as PyBOP Gaussian Parameters

        Attributes
        ------------
        len_mtx : int
            Number of basis function expansions
        GP : dict
            GP dictionary structure

        Returns
        --------
        betas_symbolic : list[pybamm.InputParameter]
            List of PyBAMM InputParameters added
        beta_parameters: dict
            Dictionary containing Beta Parameters as Gaussian distributions

        """
        betas_symbolic = []
        beta_parameters = {}
        for i in range(len_mtx):
            key_str = GP["Name"] + " Beta " + str(i)
            betas_symbolic.append(pybamm.InputParameter(key_str))
            if i == 0:
                beta_parameters[key_str] = Parameter(
                    Gaussian(GP["Constant mean"], GP["Constant standard deviation"]),
                )

            else:
                beta_parameters[key_str] = Parameter(
                    Gaussian(GP["Bi mean"], GP["Bi standard deviation"]),
                )

        self.beta_parameters = beta_parameters
        return betas_symbolic, beta_parameters

    def add_to_params(self, GP, damtxs):
        """
        Updates parameter dictionary with function, input terms
        """

        betas_function, beta_parameters = self.create_beta_inputs(len(damtxs) + 1, GP)
        self.parameter_values.update(beta_parameters)
        pybamm_function = self.add_function(
            GP["Name"],
            damtxs,
            betas_function,
            GP["Input names"],
            arguments=GP["Arguments"],
            division_arguments=GP["Division Arguments"],
            norm_bounds=GP["Normalization min-max"],
            exp=GP["exp"],
        )
        self.parameter_values.update({GP["Name"]: pybamm_function})

    def get_parameter_values(self):
        """
        returns parameter values
        """
        return self.parameter_values
