import mimic
import numpy as np

dim = 2
individuals = [mimic.Individual(np.random.rand(dim)) for _ in range(100)]
environment = mimic.Environment(mimic.benchmark.sphere())
population = mimic.Population(individuals, environment = environment)
elite = mimic.utils.population.get_elite(population)
# s_poop = mimic.utils.population.squeeze(population, (0, 3, 10))
# print(len(s_poop))

print(elite)