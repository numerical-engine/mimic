from mimic.core.Survival import Survival_meta
from mimic import utils

class genitor(Survival_meta):
    def run(self, population, mu:int, lam:int):
        assert mu > lam

        parents, offsprings = self.split(population, mu, lam)
        parents_sorted = utils.population.sort(parents)
        parents_survival = utils.population.squeeze(parents_sorted, range(mu - lam))

        return utils.population.concatenate(parents_survival, offsprings)