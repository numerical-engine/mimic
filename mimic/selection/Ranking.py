from mimic.core.Selection import Selection_meta
from mimic import utils
import numpy as np

class ranking(Selection_meta):
    """Ranking selection

    Args:
        selection_way (function): sampling way from probability distribution.
        s (float): Hyperparamter which is in (1, 2].
    Attributes:
        selection_way (function): sampling way from probability distribution.
        s (float): Hyperparamter which is in (1, 2]. If s equals to 1, this method becomes same with uniform selection. Increasing it makes selection pressure large.
    """
    def __init__(self, selection_way = utils.selection.SUS, s:float = 1.5):
        assert (s > 1.) and (s <= 2.)
        self.selection_way = selection_way
        self.s = s
    
    def get_rankscore(self, score:np.ndarray)->np.ndarray:
        """Returns ranking score

        Args:
            score (np.ndarray): Score of each individuals
        Returns:
            np.ndarray: Ranking score
        """
        N = len(score)
        ranking = np.argsort(score)
        ranking_score = np.zeros(N)
        i = N - 1
        for r in ranking:
            ranking_score[r] = i
            i -= 1
        return ranking_score
    def run(self, score:np.ndarray, selection_num:int)->tuple[int]:
        """Select individuals for parents

        Args:
            score (np.ndarray): Score of all individuals
            selection_num (int): Number of selection (i.e., mu)

        Returns:
            tuple[int]: Indice of selected individuals
        """
        ranking_score = self.get_rankscore(score)
        N = len(score)
        prob = (2. - self.s)/N + 2.*ranking_score*(self.s - 1.)/N/(N-1)

        return self.selection_way(selection_num, prob)


class ranking_exp(ranking):
    """Ranking selection of which probability is exponential style.

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
        ranking_score = self.get_rankscore(score)
        prob = 1. - np.exp(-ranking_score)
        prob /= np.sum(prob)

        return self.selection_way(selection_num, prob)
