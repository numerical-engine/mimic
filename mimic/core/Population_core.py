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
    
    def reset_score(self):
        for i in range(len(self)):
            self.individuals[i].score = None

    def shuffle(self):
        indice = np.arange(len(self))
        np.random.shuffle(indice)
        self.individuals = [self.individuals[i] for i in indice]

    @property
    def already_eval(self):
        for individual in self:
            if  individual.already_eval == False:
                return False
        return True
    @property
    def already_fit(self):
        for individual in self:
            if  individual.already_fit == False:
                return False
        return True