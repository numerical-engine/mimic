import numpy as np
from copy import deepcopy

class population:
    """The class for managing set of individual

    Args:
        individuals (list[core.Individual.individual]): List of individuals.
        generation (int): Number of generation.
        env (core.Environment.environment): Environment which we concern. If None, evaluation are skipped.
    Attributes:
        individuals (list[core.Individual.individuals]): List of individuals.
        generation (int): Number of generation.
    
    Note:
        If some individuals hasn't been evaluated, env should be set.
    """
    def __init__(self, individuals:list, generation:int = 0, *, env = None):
        self.generation = generation
        self.individuals = individuals
        # self.evaluate(env)
    
    def __len__(self)->int:
        return len(self.individuals)
    def __getitem__(self, idx):
        return self.individuals[idx]
    def __iter__(self):
        yield from self.individuals
    def copy(self):
        """Return deep copy

        Returns:
            (population): Deep copy
        """
        return type(self)(deepcopy(self.individuals), self.generation)
    
    def evaluate(self, env, force = False):
        """Evaluate individuals

        Args:
            env (core.Environment.environment): Environment we concern.
            force (bool, optional): If True, individuals are evaluated even if they has been done. Defaults to False.
        """
        def evaluate_force(env):
            assert env is not None, "set environment for evaluation"
            for idx in range(len(self)):
                f, p, s = env.get_score(self.individuals[idx], sum = False)
                self.individuals[idx].fitness = f
                self.individuals[idx].feasible = (p == 0.)
                self.individuals[idx].score = s
        
        if force or (self.individuals[0].fitness is None) or (self.individuals[0].score is None) or (self.individuals[0].feasible is None):
            evaluate_force(env)