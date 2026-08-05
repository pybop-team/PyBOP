import numpy as np
import pybamm

import FoKL
from FoKL.getKernels import sp500, bernoulli
import itertools

import pybop
from pybop.parameters import parameter

class FoKLGP:
    """
    Creates parameter functions as decomposed GPs
    options: [dict]
    """
    def __init__(self,name, options=None, parameter_values=None, kernel='Bernoulli Polynomial', twoway=True):
        GP_dict_list = self._process_options(name, options, parameter_values)
        self.evaluate_func = self._evaluate_parameter(kernel)
        self.damtx = self._create_interaction_matrix(GP_dict_list['Number of terms'], GP_dict_list['Number of inputs'], twoway)
        self.add_to_params(GP_dict_list,self.damtx)
    def __call__(self, *args):
        return self.pybamm_function(*args)

    def _process_options(self, name, options, parameter_values):
        """

        """
        default_options = {'arg_inds':None,  'exp':True, 'Number of inputs':1, 'div_arg':None, 'div_const':None,
                           'Number of terms':1, 'Constant standard deviation':0.5,'Bi mean':0, 'Bi standard deviation':0.5}
        default_options.update({'Name':name})
        if options is not None:
            default_options.update(options)
        if 'Constant mean' not in default_options:
            # If no beta constant term distribution described grab from parameters, if this is a function then user supplied
            try:
                if default_options['exp']:
                    B0_mean = np.log(parameter_values[name])

                else:
                    B0_mean = parameter_values[name]
                default_options.update({'Constant mean': B0_mean})
            except:
                raise ValueError(f'Default parameter value for {name} is not a constant, please supply an estimate')
        num_inputs = 0
        if default_options['arg_inds'] is not None:
            num_inputs += len(default_options['arg_inds'])
        if default_options['div_arg'] is not None:
            num_inputs += len(default_options['div_arg'])

        default_options['Number of inputs']=num_inputs
        self.parameter_values = parameter_values
        return default_options

    def _create_interaction_matrix(self, number_of_terms, number_of_inputs, twoway, damtx=[]):
        def perms(x):
            """Python equivalent of MATLAB perms."""
            a = np.array(np.vstack(list(itertools.permutations(x)))[::-1])
            return a

        if twoway:
            sett = 2
        else:
            sett = 1


        for ind in range(1, number_of_terms+1):
            indvec = np.zeros((number_of_inputs))
            summ = ind
            while summ:
                for j in range(0, sett):
                    indvec[j] = indvec[j] + 1
                    summ = summ - 1
                    if summ == 0:
                        break

            vecs = np.unique(perms(indvec), axis=0)

            if np.size(damtx) == 0:
                damtx = vecs
            else:
                damtx = np.append(damtx, vecs, axis=0)

        print(damtx)
        return damtx.astype(int)

    def _set_kernel(self, kernel = 'Bernoulli Polynomial'):
        if kernel == 'Cubic Splines':
            self.phis = sp500()
        elif kernel == 'Bernoulli Polynomial':
            self.phis = bernoulli()

    def _evaluate_parameter(self, kernel):
        """
        The symbolic evaluation of the decomposed GP.

        kernel: Kernel function for evaluation, must be `Cubic Splines` or `Bernoulli Polynomial`
        """
        self._set_kernel(kernel)

        if kernel == 'Cubic Splines':
            def evaluate_pybamm(
                    betas,
                    mtx,
                    inputs,
                    coeff=None):


                num_basis_terms = len(mtx)
                num_inputs = len(mtx[0])
                X_sol = []

                mtx = np.array(mtx)
                phind = []
                for i in range(num_inputs):
                    phind_temp = inputs[i] * 499
                    sett = (phind_temp == 0)
                    phind_temp = phind_temp + sett
                    r = 1 / 499  # interval of when basis function changes (i.e., when next cubic function defines spline)
                    phind.append(phind_temp - 1)

                A = [1, 2, 3]

                X_sc = [(1 - inputs[0]) ** a for a in A]

                lspace = []
                for i in range(num_inputs):
                    lspace.append(np.linspace(0, 499, 499))
                lspace = np.array(lspace)

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
                                    phi_interp = pybamm.Interpolant(lspace[0], phispace[0],
                                                                    phind[k])
                                    coeff.append(phi_interp)

                            phi *= coeff[0] + coeff[1] * X_sc[0] + coeff[2] * X_sc[1] + coeff[3] * X_sc[2]
                    X_sol.append(phi)

                X_sol_ones = betas[0]
                mean = X_sol_ones
                for i in range(len(X_sol)):
                    X_sol_betas = X_sol[i] * betas[i + 1]
                    mean += X_sol_betas

                return mean
        elif kernel == 'Bernoulli Polynomial':
            def evaluate_pybamm(betas,
                                mtx,
                                inputs,
                                coeff=None):
                """
                Pybamm Function evaluation
                betas: indexed from beta list that relates to
                """
                n = 1
                num_basis_terms = len(mtx)
                num_inputs = len(mtx[0])
                X_sol = []

                mtx = np.array(mtx)

                def bernoulli_func(phis, num, x):
                    if num > 0:
                        coeff = phis[num - 1]
                        result = coeff[0] + sum(coeff[k] * (x ** k) for k in range(1, len(coeff)))
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
            raise NotImplementedError("Kernel must be either `Cubic Splines` or `Bernoulli Polynomial`")

        return evaluate_pybamm

    def add_function(self, name, mtx, arg_inds, betas_function, exp=False, div_arg=None, div_const=None,
                     new_children=None):
        """
        Creates Parameter function specified as a GP object
        inputs:
            name: str, Name of parameter to be estimated
            arg_inds: list of int, Index of inputs to parameter function from PyBaMM
            list_index: int, Index of function hyperparameters, betas and mtx, supplied in initialization
            exp: Bool, if function should be exponential
            div_arg: list of lists of int, optional, Index of arguments to be used as div
                ex: div_arg = [[1,4],[3,2]]
                inputs to GP would be GP((1/2),(3,2))
        """

        beta_func = betas_function

        if div_arg:
            if exp:
                def pybamm_function(*args):
                    xs = []
                    for x in div_arg:
                        xs.append([args[x[0]] / args[x[1]]])
                    for x in arg_inds:
                        xs.append([args[x]])

                    res = np.exp(self.evaluate_func(beta_func, mtx, xs))
                    return res
            else:
                def pybamm_function(*args):
                    xs = []
                    for x in div_arg:
                        xs.append([args[x[0]] / args[x[1]]])
                    for x in arg_inds:
                        xs.append([args[x]])

                    res = self.evaluate_func(beta_func, mtx, xs)
                    return res
        else:
            if exp:
                def pybamm_function(*args):
                    xs = []
                    for x in arg_inds:
                        if div_const:
                            xs.append([args[x] / div_const[0]])
                        else:
                            xs.append([args[x]])

                    res = np.exp(self.evaluate_func(beta_func, mtx, xs))
                    return res
            else:
                def pybamm_function(*args):
                    xs = []
                    for x in arg_inds:
                        xs.append([args[x]])

                    res = self.evaluate_func(beta_func, mtx, xs)
                    return res
        if type(self.parameter_values[name]) is not float:
            function_args = self.parameter_values[name].__code__.co_varnames
            function_args_mod = []
            if div_arg is not None:
                for x in div_arg:
                    function_args_mod.append(function_args[x[0]] + str('/') + function_args[x[1]])
            for x in arg_inds:
                function_args_mod.append(function_args[x])
            print(f"GP function created for {name} \n inputs are {function_args_mod}")
        self.pybamm_function = pybamm_function
        return pybamm_function


    def create_beta_inputs(self,len_mtx, GP):
        """
        Generates Input Variables
        """
        betas_symbolic = []
        beta_parameters = {}
        for i in range(len_mtx):
            key_str = GP['Name'] + ' Beta ' + str(i)
            betas_symbolic.append(pybamm.InputParameter(key_str))
            if i == 0:
                beta_parameters[key_str] = pybop.Parameter(
                    distribution=pybop.Gaussian(GP['Constant mean'], GP['Constant standard deviation']),
                )

            else:
                beta_parameters[key_str] = pybop.Parameter(
                    distribution=pybop.Gaussian(GP['Bi mean'], GP['Bi standard deviation']),
                )

        self.beta_parameters = beta_parameters
        return betas_symbolic, beta_parameters

    def add_to_params(self, GP, damtxs):

        betas_function, beta_parameters = self.create_beta_inputs(len(damtxs) + 1, GP)
        self.parameter_values.update(beta_parameters)
        pybamm_function = self.add_function(GP['Name'], damtxs, GP['arg_inds'], betas_function, exp=GP['exp'], div_arg=GP['div_arg'],
                       div_const=GP['div_const'])
        self.parameter_values.update({GP['Name']:pybamm_function})

    def get_parameter_values(self):
        return self.parameter_values


