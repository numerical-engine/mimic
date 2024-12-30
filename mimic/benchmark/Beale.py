import numpy as np
from mimic.core.Function_core import Function

class Beale(Function):
    def forward(self, individual)->float:
        assert individual.dim() == 2, f"dim equals to {individual.dim()}"
        x, y = individual.x

        return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2