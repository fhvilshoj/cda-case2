import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV


from sklearn.linear_model import LogisticRegression as LR

class LogReg:
    def __init__(self):
        "docstring"
        # C_range = np.logspace(-2, 3, 6)
        C_range = np.logspace(-2, 3, 1)
        
        max_iter_range = [25, 50, 100, 200, 400]
        
        self.param_grid = dict(max_iter=max_iter_range) # C=C_range, 
        self.cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        self.grid = GridSearchCV(LR(), param_grid=self.param_grid, cv=self.cv, verbose=True)
