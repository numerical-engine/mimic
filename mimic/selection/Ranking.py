from mimic.core.Selection_core import Selection
from mimic import utils
import numpy as np

class ranking(Selection):
    def __init__(self, selection_way = utils.Selection.roulette_wheel, s:float = 1.5):
        assert (s > 1.) and (s <= 2.)
        self.selection_way = selection_way
        self.s = s
    
    def get_rankscore(self, score:np.ndarray)->np.ndarray:
        N = len(score)
        ranking = np.argsort(score)
        ranking_score = np.zeros(N)
        i = N - 1
        for r in ranking:
            ranking_score[r] = i
            i -= 1
        return ranking_score
    def run(self, score:np.ndarray, selection_num:int)->tuple[int]:
        ranking_score = self.get_rankscore(score)
        N = len(score)
        prob = (2. - self.s)/N + 2.*ranking_score*(self.s - 1.)/N/(N-1)

        return self.selection_way(selection_num, prob)