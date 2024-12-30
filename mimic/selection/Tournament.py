from mimic.core.Selection_core import Selection
import numpy as np


class tournament(Selection):
    def __init__(self, k:int = 2, replace:bool = True):
        self.k = k
        self.replace = replace
    
    def run(self, score:np.ndarray, selection_num:int)->tuple:
        if not self.replace:
            assert len(score) >= (self.k*selection_num)
        players = np.arange(len(score))
        indice = []

        for _ in range(selection_num):
            if self.replace:
                player_indice = np.random.choice(players, self.k, replace = False)
            else:
                player_indice = np.random.choice(np.setdiff1d(players, indice), self.k, replace = False)
            player_score = score[player_indice]
            winner_idx = player_indice[np.argmin(player_score)]
            indice.append(winner_idx)
        
        return tuple(indice)