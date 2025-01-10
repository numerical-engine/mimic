import numpy as np
from mimic import utils

class Survival:
    def __call__(self, parents, offsprings, environment = None, reset_score:bool = True):
        mu = len(parents)
        lam = len(offsprings)

        population = utils.Population.concatenate(parents, offsprings)

        if reset_score:
            population.reset_score()
        
        if not population.already_eval:
            environment.set_score(population)
        
        population_new = self.run(population, mu, lam)
        assert len(population_new) == mu

        return population_new
    
    def split(self, population, mu:int, lam:int):
        assert len(population) == (mu+lam)
        indice = tuple(i for i in range(mu+lam))
        parents = utils.Population.squeeze(population, indice[:mu])
        offsprings = utils.Population.squeeze(population, indice[mu:])

        return parents, offsprings
    
    def run(self, population, mu:int, lam:int):
        raise NotImplementedError