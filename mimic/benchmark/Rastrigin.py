import numpy as np
from mimic.core import Function_meta

class Rastrigin(Function_meta):
    """Rastrigin benchmark function

    Args:
        n (int): Dimension of solution
    Attributes:
        n (int): Dimension of solution
    """
    def __init__(self, n:int):
        assert n >= 1, f"n should be greater than 1"
        self.n = n
    def forward(self, individual)->float:
        """Output objective function value
        
        Args:
            individual (core.Individual.Individual): Individual
        Returns:
            float: Objective function value.
        """
        assert self.n == individual.dim()
        f = 10.*self.n
        for x in individual.x:
            f += (x**2 - 10.*np.cos(2.*np.pi*x))
        return f