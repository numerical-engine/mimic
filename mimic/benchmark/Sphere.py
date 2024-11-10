import numpy as np
from typing import Union
from mimic.core import function_meta

class sphere(function_meta):
    def forward(self, x:np.ndarray)->Union[float, np.ndarray]:
        """Output objective function value
        Args:
            x (np.ndarray): Solution
        Returns:
            (float, np.ndarray): Objective function value. If x.shape equals to (N, dim), return np.ndarray which shape equals to (N, ).
        
        **Note**  
            * x.shape should be (N, dim) or (dim, ).
        """
        return np.sum(x**2, axis = 1)