import os
import numpy
import matplotlib

class QuickPlot(item):
    """
    
    Class to generate the quick plots associated with PRISM. 

    Plots
    --------------
    Observability
    if method == parameterisation 
        
        Comparison of fitting data with optimised forward model
    
    elseif method == optimisation
        
        Pareto front
        Alternative solutions
        Initial value compared to optimal 

    """

    def __init__(self):
        self.name = "Quick Plot"


