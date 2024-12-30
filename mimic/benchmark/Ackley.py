import numpy as np
from mimic.core.Function_core import Function

class Ackley(Function):
    def forward(self, individual)->float:
        assert individual.dim() == 2, f"dim equals to {individual.dim()}"
        x, y = individual.x