from mimic.core.Crossover_core import Crossover
import numpy as np

class blx_alpha(Crossover):
    def __init__(self, offspring_num:int = 2, alpha:float = 0.5):
        super().__init__(offspring_num, 2)
        self.alpha = alpha
    
    def run(self, x_pair:tuple[np.ndarray])->list[np.ndarray]:
        x1, x2 = x_pair
        min_values = np.minimum(x1, x2)
        max_values = np.maximum(x1, x2)
        length = max_values - min_values

        x_offs = [np.random.uniform(min_values - self.alpha*length, max_values + self.alpha*length) for _ in range(self.offspring_num)]

        return x_offs