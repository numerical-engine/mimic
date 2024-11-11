import mimic
import numpy as np

x = np.array([1.,2.,3.])
att = {"sigma":np.ones(1)}
ind = mimic.core.Individual(x, att = att)
ind2 = ind.copy()
print(ind2.sigma)