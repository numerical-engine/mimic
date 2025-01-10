import numpy as np

class Crossover:
    def __init__(self, offspring_num:int = 2, parent_num:int = 2):
        self.parent_num = parent_num
        self.offspring_num = offspring_num

    def make_pairs(self, parents)->list[list]:
        assert (len(parents) % self.parent_num) == 0
        indice = np.arange(len(parents)).astype(int)
        np.random.shuffle(indice)

        indice = indice.reshape((-1, self.parent_num))
        return [[parents[i].copy() for i in idx] for idx in indice]
    
    def __call__(self, parents):
        pair_list = self.make_pairs(parents) #list[list[Individual]]
        assert len(pair_list[0]) == self.parent_num
        offsprings = [] #list[Individual]
        for pair in pair_list:
            offsprings += self._run(pair)
        
        offsprings = type(parents)(offsprings, parents.generation)

        return offsprings
    
    def _run(self, pair:list)->list:
        x_pair = tuple(individual.x for individual in pair)
        x_offs = self.run(x_pair)
        assert len(x_offs) == self.offspring_num
        
        offspring = [type(pair[0])(x) for x in x_offs]
        return offspring
    
    def run(self, x_pair:tuple[np.ndarray])->list[np.ndarray]:
        raise NotImplementedError