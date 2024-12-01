import numpy as np

class Mutation_meta:
    """Class for mutation and updating population

    Attributes:
        att_keys (list[str]): list of names which should be defined in att of individuals
    """
    def __init__(self, att_keys = []):
        self.att_keys = att_keys
    def __call__(self, population, **kwargs):
        ind_att_keys = population[0].att_keys
        for key in self.att_keys:
            assert key in ind_att_keys, f"{key} doesn't contain in Individual"
        population_new = population.copy()

        population_new.individuals = [self.run(individual, **kwargs) for individual in population_new]
        return population_new
    
    def run(self, individual, **kwargs):
        """Return a mutated individual

        Args:
            individual (core.Individual.Individual): Individual
        Raises:
            NotImplementedError: This method should be overwritten
        Returns:
            core.Individual.Individual: Mutated individual
        Note:
            kwargs are for inheriting class
        """
        raise NotImplementedError