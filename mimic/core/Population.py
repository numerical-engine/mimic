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
    
    Note:
        If some individuals hasn't been evaluated, env should be set.
    """
    def __init__(self, individuals:list, generation:int = 0, *, environment = None):
        self.generation = generation
        self.individuals = individuals
        self.evaluate(environment)
    
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
    
    def evaluate(self, environment, force = False):
        """Evaluate individuals

        Args:
            environment (core.Environment.Environment): Environment we concern.
            force (bool, optional): If True, individuals are evaluated even if they has been done. Defaults to False.
        """
        def evaluate_force(environment):
            assert environment is not None, "set environment for evaluation"
            for idx in range(len(self)):
                f, p, s = environment.get_score(self.individuals[idx], sum = False)
                self.individuals[idx].fitness = f
                self.individuals[idx].feasible = (p == 0.)
                self.individuals[idx].score = s
        
        if force or (self.individuals[0].fitness is None) or (self.individuals[0].score is None) or (self.individuals[0].feasible is None):
            evaluate_force(environment)