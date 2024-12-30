from mimic.core.Crossover_core import Crossover
import numpy as np

class uniform(Crossover):
    def __init__(self, prob:float = 0.5):
        super().__init__(2, 2)
        self.prob = prob
    
    def run(self, x_pair:tuple[np.ndarray])->list[np.ndarray]:
        x1, x2 = x_pair
        mask = (np.random.rand(len(x1)) < self.prob).astype(float)
        return [mask*x1 + (1. - mask)*x2, mask*x2 + (1. - mask)*x1]