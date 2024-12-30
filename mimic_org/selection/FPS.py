from mimic.core.Selection import Selection_meta
from mimic import utils
import numpy as np

class fps(Selection_meta):
    """FPS selection

    Args:
        selection_way (function): sampling way from probability distribution.
        window (float): Window value.
    Attributes:
        selection_way (function): sampling way from probability distribution.
        window (float): Window value.
    """
    def __init__(self, selection_way = utils.selection.SUS, window:float = 0.):
        self.selection_way = selection_way
        self.window = window
    def run(self, score:np.ndarray, selection_num:int)->tuple[int]:
        """Select individuals for parents

        Args:
            score (np.ndarray): Score of all individuals
            selection_num (int): Number of selection (i.e., mu)

        Returns:
            tuple[int]: Indice of selected individuals
        """
        max_score = np.max(score)
        fixed_score = max_score - score + self.window
        prob = fixed_score/np.sum(fixed_score)

        return self.selection_way(selection_num, prob)