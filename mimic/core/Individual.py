import numpy as np
from copy import deepcopy
import sys

class Individual:
    """Class for a individual.

    Args:
        x (np.ndarray): Solution. Shape should be (dim, ).
        fitness (float): Objective function value. If None, this solution hasn't been evaluated.
        score (float): Socre. If None, this solution hasn't been evaluated.
        feasible (bool): Feasibility of this solution. If None, this solution hasn't been evaluated.
        age (int): Age
    Attributes:
        x (np.ndarray): Solution. Shape should be (dim, ).
        fitness (float): Objective function value. If None, this solution hasn't been evaluated.
        score (float): Socre. If None, this solution hasn't been evaluated.
        feasible (bool): Feasibility of this solution. If None, this solution hasn't been evaluated.
        age (int): Age
        att_keys (list[str]): List of names given from att.
        att_dims (list[int]): List of dimension given from att.
        key (np.ndarray): Suplemental variables given from att. The name "key" is same with att_keys[i].
    """
    def __init__(self, x:np.ndarray, fitness:float = None, score:float = None, feasible:bool = None, age:int = 0, *, att:dict = {}):
        assert len(x.shape) == 1, f"Individual doesn't support batch soluion. len(x.shape) should be (dim, ), but got {x.shape}"
        self.x = x
        self.age = age
        self.fitness = fitness
        self.score = score
        self.feasible = feasible

        ##### load **att as a Attributes
        self.att_keys = []
        self.att_dims = []
        for key in att.keys():
            assert isinstance(att[key], np.ndarray), f"{key} should be numpy"
            assert len(att[key].shape) == 1, f"{key} should be 1st order tensor, but shape is {att[key].shape}"
            self.att_keys.append(key); self.att_dims.append(len(att[key]))
            self.__dict__[key] = att[key]
    
    def __call__(self)->tuple:
        return self.age, self.x, self.fitness
    def __str__(self)->str:
        return str(self())
    def __repr__(self)->str:
        return self.__str__()
    def att_dict(self)->dict:
        """Return dictionary about additional variables

        Returns:
            dict: Dictionary about additional variables which is same with att of _init_.
        """
        return {key : self.__dict__[key] for key in self.att_keys}
    def dim(self, include_att:bool = False)->int:
        """Dimension of this individual

        Args:
            include_att (bool, optional): Including additional variables or not. Defaults to False.
        Returns:
            int: Dimension
        """
        return len(self.X) + np.sum(self.att_dims) if include_att else len(self.X)
    
    def copy(self):
        """Return deep copy

        Returns:
            core.Individual.Individual: Deep copy
        """
        x = deepcopy(self.x); att = deepcopy(self.att_dict())
        return type(self)(x, self.fitness, self.score, self.feasible, self.age, att = att)
    
    def __add__(self, another):
        X = self.x + another.x
        att = {key : self.__dict__[key] + another.__dict__[key] for key in self.att_keys}
        return type(self)(X, age = 0, att = att)
    
    def __mul__(self, another):
        X = another*self.x
        att = {key : another*self.__dict__[key] for key in self.att_keys}
        return type(self)(X, age = 0, att = att)
    
    def __rmul__(self, another):
        return self.__mul__(another)