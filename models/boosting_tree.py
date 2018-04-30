import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

class AdaBoost:
    def __init__(self):
        """docstring 
        
        """
        
        n_estimators_range = np.array([50]) # np.arange(25, 101, 25) # [25, 50, 75, 100]
        leaning_rate_range = np.array([1.])# np.logspace(-2, 2, 5) # [1.e-02, 1.e-01, 1.e-00, 1.e+01, 1.e+02]

        self.learning_rate_range = leaning_rate_range
        
        self.param_grid = dict(n_estimators=n_estimators_range, learning_rate=leaning_rate_range)
        self.cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        self.grid = GridSearchCV(AdaBoostClassifier(), param_grid=self.param_grid, cv=self.cv, verbose=True)

    def set_n_esimators(self, n_estimators):
        self.param_grid = dict(learning_rate=self.leaning_rate_range)
        self.cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        self.grid = GridSearchCV(AdaBoostClassifier(n_estimators=n_estimators),
                                 param_grid=self.param_grid,
                                 cv=self.cv,
                                 verbose=True)
