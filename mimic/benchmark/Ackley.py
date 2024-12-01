import numpy as np
from mimic.core import Function_meta

class Ackley(Function_meta):
    """Beale benchmark function
    """
    def forward(self, individual)->float:
        """Output objective function value
        
        Args:
            individual (core.Individual.Individual): Individual
        Returns:
            float: Objective function value.
        """
        assert individual.dim() == 2, f"dim equals to {individual.dim()}"
        x, y = individual.x