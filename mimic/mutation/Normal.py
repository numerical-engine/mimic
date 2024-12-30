import numpy as np
from mimic.core.Mutation_core import Mutation
from typing import Union

class normal(Mutation):
    def __init__(self, sigma:Union[float, np.ndarray], dim:int = None):
        super().__init__()
        self.sigma = np.ones(dim)*sigma if isinstance(sigma, float) else np.asarray(sigma)
        assert np.all(self.sigma > 0.)
    
    def run(self, individual):
        individual.x += np.random.normal(scale = self.sigma)
        return individual