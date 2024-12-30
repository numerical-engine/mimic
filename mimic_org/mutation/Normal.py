import numpy as np
from mimic.core.Mutation import Mutation_meta
from typing import Union

class normal(Mutation_meta):
    """Mutation of normal distribution

    Args:
        sigma (Union[float, np.ndarray]): STD. If float, use sigma to all elements.
        dim (int): Dimension of solution (i.e., len(individual.x))
    Attributes:
        sigma (np.ndarray): STD of each element.
    """
    def __init__(self, sigma:Union[float, np.ndarray], dim:int = None):
        super().__init__()
        self.sigma = np.ones(dim)*sigma if isinstance(sigma, float) else np.asarray(sigma)
        assert np.all(self.sigma > 0.)
    
    def run(self, individual):
        """Return a mutated individual

        Args:
            individual (core.Individual.Individual): Individual
        Returns:
            core.Individual.Individual: Mutated individual
        """
        individual.x += np.random.normal(scale = self.sigma)
        return individual

class adaptive_normal(Mutation_meta):
    """Mutation of adaptive variance normal distribution

    Args:
        sigma_epsilon (Union[float, np.ndarray]): Lower limit of STD. If float, use sigma to all elements.
        dim (int): Dimension of solution (i.e., len(individual.x))
    Attributes:
        sigma_epsilon (np.ndarray): lower limit of STD of each element.
    """
    def __init__(self, sigma_epsilon:np.ndarray, dim:int = None):
        super().__init__(["sigma"])
        self.sigma_epsilon = np.ones(dim)*sigma_epsilon if isinstance(sigma_epsilon, float) else np.asarray(sigma_epsilon)
        assert np.all(self.sigma_epsilon > 0.)
    
    def run(self, individual, *, pop_size:int, alpha:float = 1.):
        """Return a mutated individual

        Args:
            individual (core.Individual.Individual): Individual
            pop_size (int): Population size.
            alpha (float): Hyper parameter of learning rate.
        Returns:
            core.Individual.Individual: Mutated individual
        """
        tau1 = alpha/np.sqrt(2.*pop_size)
        tau2 = alpha/np.sqrt(2.*np.sqrt(pop_size))
        rand1 = np.random.normal()
        rand2 = np.random.normal(size = len(individual.sigma))

        sigma_new = individual.sigma*np.exp(tau1*rand1 + tau2*rand2)
        sigma_new = np.maximum(sigma_new, self.sigma_epsilon)

        rand = np.random.normal(size = len(individual.sigma))
        individual.x += sigma_new*rand
        individual.sigma = sigma_new

        return individual