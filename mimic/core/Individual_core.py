import numpy as np
from copy import deepcopy

class Individual_core:
    def __init__(self, x:np.ndarray, fitness = None, score:float = None, penalty = None, age:int = 0):
        assert len(x.shape) == 1, f"Individual doesn't support batch soluion. len(x.shape) should be (dim, ), but got {x.shape}"
        self.x = x
        self.age = age
        self.fitness = fitness
        self.score = score
        self.penalty = penalty
    
    def __call__(self)->tuple:
        return self.age, self.x, self.fitness, self.penalty, self.score
    def __str__(self)->str:
        return str(self())
    def __repr__(self)->str:
        return self.__str__()
    @property
    def dim(self)->int:
        return len(self.x)
    @property
    def feasible(self)->bool:
        raise NotImplementedError
    
    @property
    def already_eval(self):
        if (self.fitness is None) or (self.penalty is None) or (self.score is None):
            return False
        else:
            return True

    def copy(self):
        x = deepcopy(self.x)
        fitness = deepcopy(self.fitness)
        score = deepcopy(self.score)
        penalty = deepcopy(self.penalty)

        return type(self)(x = x, fitness = fitness, score = score, penalty = penalty, age = self.age)
    
    def __add__(self, another):
        x = self.x + another.x
        return type(self)(x, age = 0)
    
    def __mul__(self, another):
        x = another*self.x
        return type(self)(x, age = 0)
    
    def __rmul__(self, another):
        return self.__mul__(another)