import pybamm

class Simulation(pybamm.simulation.Simulation):
    """
    
    This class constructs the PyBOP Simulation class

    Parameters:
    ================
   

    """

    def __init__(self, *args):
       super(Simulation, self).__init__(*args)


    def optimize(self):
        """
        
        Optimize function for the give optimisation problem.

        """