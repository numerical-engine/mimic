import numpy as np
from typing import Union

class function_meta:
    """Abstruct class for any function.

    Note:
        Any function should be defined by classes inheriting function_meta, because forward method isn't implemented.
        Codes in `./mimic/benchmark/` will help you.
    """
    def __call__(self, x:np.ndarray)->Union[float, np.ndarray, np.float64]:
        assert len(x.shape) <= 2, f"x.shape should be (N, dim) or (dim, ). But got {x.shape}"
        return self.forward(x)
    def forward(self, x:np.ndarray)->Union[float, np.ndarray]:
        """Output function value

        Args:
            x (np.ndarray): Solution which shape equals to (N, dim) or (N, ).
        Returns:
            (float, np.ndarray): Function value. If x.shape equals to (N, dim), return np.ndarray which shape equals to (N, ).
        """
        raise NotImplementedError


class penalty_function(function_meta):
    """Inequality penalty function

    Args:
        pfunc (function): Penalty function of which input and output type should be same with forward method.
        weight (float): Weight parameter. Defaults to 1.0.
    Attributes:
        pfunc (function): Penalty function of which input and output type should be same with forward method.
        weight (float): Weight parameter. Defaults to 1.0.
    Note:
        MIMIC can adapt only **pfunc(x) <= 0** like constraint
    """
    def __init__(self, pfunc, weight:float = 1.):
        self.weight = weight
        self.pfunc = pfunc
    
    def forward(self, x:np.ndarray)->Union[float, np.ndarray]:
        """Output penalty function value

        Args:
            x (np.ndarray): Solution which shape equals to (N, dim) or (N, ).
        Returns:
            (float, np.ndarray): Function value. If x.shape equals to (N, dim), return np.ndarray which shape equals to (N, ).
        """
        penalty_values = self.weight*self.pfunc(x)
        #####If penalty_values < 0, set to 0. because x is feasible.
        return np.maximum(penalty_values, 0.)