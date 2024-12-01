from mimic.core.Selection import Selection_meta
from mimic import utils
import numpy as np

class uniform(Selection_meta):
    """Uniform selection

    Args:
        selection_way (function): sampling way from probability distribution.
    Attributes:
        selection_way (function): sampling way from probability distribution.
    """
    def __init__(self, selection_way = utils.selection.SUS):
        self.selection_way = selection_way
    
    def run(self, score:np.ndarray, selection_num:int)->tuple[int]:
        """Select individuals for parents

        Args:
            score (np.ndarray): Score of all individuals
            selection_num (int): Number of selection (i.e., mu)

        Returns:
            tuple[int]: Indice of selected individuals
        """
        prob = np.ones(len(score))/len(score)
        return self.selection_way(selection_num, prob)