import numpy as np
from mimic.core import Function_meta

class sphere(Function_meta):
    """Sphere benchmark function
    """
    def forward(self, individual)->float:
        """Output objective function value
        
        Args:
            individual (core.Individual.Individual): Individual
        Returns:
            float: Objective function value.
        """
        return np.sum(individual.x**2)