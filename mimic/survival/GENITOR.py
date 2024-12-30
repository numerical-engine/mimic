from mimic.core.Survival_core import Survival
from mimic import utils

class genitor(Survival):
    def run(self, population, mu:int, lam:int):
        assert mu > lam

        parents, offsprings = self.split(population, mu, lam)
        parents_sorted = utils.population.sort(parents)
        parents_survival = utils.population.squeeze(parents_sorted, range(mu - lam))

        return utils.population.concatenate(parents_survival, offsprings)