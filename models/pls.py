import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.cross_decomposition import PLSRegression

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

class PLS:
    def __init__(self):
        """docstring 
        
        """
        
        n_components_range = np.logspace(2, 10, 5, base=2).astype(np.int32) #[   4.,   16.,   64.,  256., 1024.]

        self.param_grid = dict(n_components=n_components_range)
        self.cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        self.grid = GridSearchCV(PLSRegression(), param_grid=self.param_grid, cv=self.cv, verbose=True)

