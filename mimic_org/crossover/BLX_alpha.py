from mimic.core.Crossover import Crossover_meta
import numpy as np

class blx_alpha(Crossover_meta):
    """BLX_alpha crossover

    Args:
        alpha (float): Hyper parameter
        offspring_num (int): Number of offsprings given from one crossover
    Attributes:
        parent_num (int): Number of parents for crossover.
        offspring_num (int): Number of offsprings given from one crossover.
        alpha (float): Hyper parameter
    """
    def __init__(self, offspring_num:int = 2, alpha:float = 0.5):
        super().__init__(offspring_num, 2)
        self.alpha = alpha
    
    def run(self, x_pair:tuple[np.ndarray])->list[np.ndarray]:
        """Generate variables for offspring

        Args:
            x_pair (tuple[np.ndarray]): tuple of variables with respect to pair
        Returns:
            list[np.ndarray]: list of variables with respect to offspring.
        """
        x1, x2 = x_pair
        min_values = np.minimum(x1, x2)
        max_values = np.maximum(x1, x2)
        length = max_values - min_values

        x_offs = [np.random.uniform(min_values - self.alpha*length, max_values + self.alpha*length) for _ in range(self.offspring_num)]

        return x_offs