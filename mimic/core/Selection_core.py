import numpy as np
from mimic import utils

class Selection:
    def __call__(self, population, selection_num:int):
        assert population.already_eval
        score = np.array([individual.score for individual in population])
        indice = self.run(score, selection_num) #tuple[int]
        assert len(indice) == selection_num
        
        selected_population = utils.Population.squeeze(population, indice)
        return selected_population
    
    def run(self, score:np.ndarray, selection_num:int)->tuple[int]:
        raise NotImplementedError