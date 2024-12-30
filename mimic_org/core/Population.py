import numpy as np
from copy import deepcopy

class Population:
    """The class for managing set of individual

    Args:
        individuals (list[core.Individual.Individual]): List of individuals.
        generation (int): Number of generation.
        env (core.Environment.Environment): Environment which we concern. If None, evaluation are skipped.
    Attributes:
        individuals (list[core.Individual.Individual]): List of individuals.
        generation (int): Number of generation.
    """
    def __init__(self, individuals:list, generation:int = 0):
        assert isinstance(generation, int)
        self.generation = generation
        self.individuals = individuals
    
    def __len__(self)->int:
        return len(self.individuals)
    def __getitem__(self, idx):
        return self.individuals[idx]
    def __iter__(self):
        yield from self.individuals
    def copy(self):
        """Return deep copy

        Returns:
            Population: Deep copy
        """
        return type(self)(deepcopy(self.individuals), self.generation)
    
    @property
    def already_eval(self):
        for individual in self:
            if  individual.already_eval == False:
                return False
        return True