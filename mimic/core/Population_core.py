import numpy as np
from copy import deepcopy

class Population:
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
        return type(self)(deepcopy(self.individuals), self.generation)
    
    @property
    def already_eval(self):
        for individual in self:
            if  individual.already_eval == False:
                return False
        return True