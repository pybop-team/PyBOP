import pybop

class BaseOptimisation:
    """
    
    Base class for the optimisation methods.
    
    """

    def __init__(self, Simulation):

        """

        Initialise and name class.
        
        """
        self.name = "Base Optimisation"
        self.Simulation = Simulation.copy()



    # def NelderMead(self, fun, x0, options):
    #     """
    #     PyBOP optimiser using Nelder-Mead.
    #     """
    #     res = scipy.optimize(fun, x0, method='nelder-mead', 
    #     options={'xatol': 1e-8, 'disp': True})