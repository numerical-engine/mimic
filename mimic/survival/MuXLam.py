from mimic.core.Survival import Survival_meta
from mimic import utils

class mu_plus_lam(Survival_meta):
    def run(self, population, mu:int, lam:int):
        population_sorted = utils.population.sort(population)

        return utils.population.squeeze(population_sorted, range(mu))

class mu_to_lam(Survival_meta):
    def run(self, population, mu:int, lam:int):
        assert lam > mu

        parents, offsprings = self.split(population, mu, lam)
        offsprings_sorted = utils.population.sort(offsprings)

        return utils.population.squeeze(offsprings_sorted, range(mu))