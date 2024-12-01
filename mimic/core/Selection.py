import numpy as np
from mimic import utils

class Selection_meta:
    """Class for selection and updating population

    Note:
        This class should be inherited, because method **run** hasn't been defined. Codes in `mimic/selction/` will help you.
    """
    def __call__(self, population, selection_num:int):
        assert population.already_eval
        score = np.array([individual.score for individual in population])
        indice = self.run(score, selection_num) #tuple[int]
        assert len(indice) == selection_num
        
        selected_population = utils.population.squeeze(population, indice)
        return selected_population
    
    def run(self, score:np.ndarray, selection_num:int)->tuple[int]:
        """select individuals for parents

        Args:
            score (np.ndarray): score of all individuals
            selection_num (int): number of selection (i.e., mu)
        Raises:
            NotImplementedError: This method should be overwritten.
        Returns:
            tuple[int]: indice of selected individuals
        """
        raise NotImplementedError