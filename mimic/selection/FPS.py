from mimic.core.Selection_core import Selection
from mimic import utils
import numpy as np

class fps(Selection):
    def __init__(self, selection_way = utils.Selection.roulette_wheel, window:float = 0.):
        self.selection_way = selection_way
        self.window = window
    def run(self, score:np.ndarray, selection_num:int)->tuple[int]:
        max_score = np.max(score)
        fixed_score = max_score - score + self.window
        prob = fixed_score/np.sum(fixed_score)

        return self.selection_way(selection_num, prob)