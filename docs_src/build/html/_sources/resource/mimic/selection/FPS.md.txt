### Fitness proportional selection
```{eval-rst}  
.. automodule:: mimic.selection.FPS
   :members:
   :undoc-members:
   :show-inheritance:
```
```{note}
Since optimization = minimization in in this module, score of each individuals are transformed to
````{math}
s' = {\rm max}(s) - s + w, 
````
where w is window.
```