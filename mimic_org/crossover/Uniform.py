from mimic.core.Crossover import Crossover_meta
import numpy as np

class uniform(Crossover_meta):
    """Uniform crossover

    Args:
        prob (float): Probability about cross over for each element
    Attributes:
        parent_num (int)=2: Number of parents for crossover.
        offspring_num (int)=2: Number of offsprings given from one crossover.
        prob (float): Probability about cross over for each element.
    """
    def __init__(self, prob:float = 0.5):
        super().__init__(2, 2)
        self.prob = prob
    
    def run(self, x_pair:tuple[np.ndarray])->list[np.ndarray]:
        """Generate variables for offspring

        Args:
            x_pair (tuple[np.ndarray]): tuple of variables with respect to pair
        Returns:
            list[np.ndarray]: list of variables with respect to offspring.
        """
        x1, x2 = x_pair
        mask = (np.random.rand(len(x1)) < self.prob).astype(float)
        return [mask*x1 + (1. - mask)*x2, mask*x2 + (1. - mask)*x1]