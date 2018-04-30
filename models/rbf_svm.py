import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

class RbfSvm:
    def __init__(self):
        """docstring 
        I Extensive search for c
        I λ less crucial, try different value
        """
        
        C_range = np.logspace(-1, 3, 5) # [1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03]
        gamma_range = np.logspace(-5, 3, 3) # [1.e-05, 1.e-01, 1.e+03]

        self.param_grid = dict(gamma=gamma_range, C=C_range)
        self.cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        self.grid = GridSearchCV(SVC(max_iter=200), param_grid=self.param_grid, cv=self.cv, verbose=True)

        # # C_range = np.logspace(-1, 3, 5) # [1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03]
        # max_iter_range = [25, 50, 100, 200, 400]

        # self.param_grid = dict(max_iter=max_iter_range)
        # self.cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        # self.grid = GridSearchCV(SVC(), param_grid=self.param_grid, cv=self.cv, verbose=True)

