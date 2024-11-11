import numpy as np
from mimic.core import Function_meta

class Rosenbrock(Function_meta):
    """Rosenbrock benchmark function

    Args:
        n (int): Parameter which is greater than 2.
    """
    def __init__(self, n:int):
        assert n >= 2., f"n should be greater than 2"
        self.n = n
    def forward(self, individual)->float:
        """Output objective function value
        
        Args:
            individual (core.Individual.Individual): Individual
        Returns:
            float: Objective function value.
        """
        f = 0.
        x = individual.x
        for i in range(len(x)-1):
            f += 100.*(x[i+1]-x[i]**2)**2 + (1.-x[i])**2
        return f