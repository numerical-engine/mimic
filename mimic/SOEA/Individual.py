import numpy as np
from mimic.core.Individual_core import Individual_core

class Individual(Individual_core):
    def __init__(self, x:np.ndarray, fitness:float = None, score:float = None, penalty:float = None, age:int = 0):
        super().__init__(x, fitness, score, penalty, age)
    
    @property
    def feasible(self)->bool:
        return (self.penalty == 0.)