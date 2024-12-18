import numpy as np
from mimic import utils

class Survival_meta:
    def __call__(self, parents, offsprings, environment = None):
        mu = len(parents)
        lam = len(offsprings)

        population = utils.population.concatenate(parents, offsprings)
        if not population.already_eval:
            environment.set_score(population)
        
        population_new = self.run(population, mu, lam)
        assert len(population_new) == mu

        return population_new
     
    def split(self, population, mu:int, lam:int):
        assert len(population) == (mu+lam)
        indice = tuple(i for i in range(mu+lam))
        parents = utils.population.squeeze(population, indice[:mu])
        offsprings = utils.population.squeeze(population, indice[mu:])

        return parents, offsprings
    
    def run(self, population, mu:int, lam:int):
        raise NotImplementedError