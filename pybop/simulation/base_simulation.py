import pybamm.simulation.Simulation as pybamm_simulation

class BaseSimulation(pybamm_simulation):
    """
    
    This class solves the optimisation / estimation problem. 

    Parameters:
    ================
    pybamm_simulation: argument for PyBaMM simulation that will be updated.

    """

    def __init__(self):
       """
       Initialise and name class
       """ 

       self.name = "Base Simulation"

    def Simulation(self, simulation, optimise, cost, data):
        """
        
        

        """