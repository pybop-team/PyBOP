import botorch
import scipy
import numpy 

class BaseOptimisation():
    """
    
    Base class for the optimisation methods.
    
    """

    def __init__(self):

        """

        Initialise and name class.
        
        """
        self.name = "Base Optimisation"



    def NelderMead(self, fun, x0, options):
        """
        PRISM optimiser using Nelder-Mead.
        """
        res = scipy.optimize(fun, x0, method='nelder-mead', 
        options={'xatol': 1e-8, 'disp': True})