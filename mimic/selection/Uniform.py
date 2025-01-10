from mimic.core.Selection_core import Selection
from mimic import utils
import numpy as np

class uniform(Selection):
    def __init__(self, selection_way = utils.Selection.roulette_wheel):
        self.selection_way = selection_way
    
    def run(self, score:np.ndarray, selection_num:int)->tuple[int]:
        prob = np.ones(len(score))/len(score)
        return self.selection_way(selection_num, prob)