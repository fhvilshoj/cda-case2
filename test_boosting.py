import sys
sys.path.append('.')
import os
import os.path
import time
import datetime

import numpy as np
import argparse

import pickle

from data_feed import Reader
from models.boosting_tree import AdaBoost as AB

def main(config):
    feed = Reader()
    
    train_images, train_labels = feed.data()
    train_images = train_images.reshape((-1, np.prod(train_images.shape[1:])))

    model_name = 'Ada_Boost_DT'
    n_estimators = [25, 50, 75, 100, 200]
    model = AB()

    for n_estimator in n_estimators:
        
        model.set_n_estimators(n_estimator)
        model.grid.fit(train_images, train_labels)

        print("++++++++++++++++++"*2)
        print(model.grid)
        print("++++++++++++++++++"*2)
        print(model.grid.cv_results_)

        if not os.path.exists('../results'):
            os.makedirs('../results')

        ts = time.time()
        t_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M')
        file_name = os.path.join('../results', f'{model_name}_{n_estimator:03}_{t_str}.pkl')
        print('Writing result to {file_name}')
        
        with open(file_name, 'wb') as f:
            pickle.dump(model.grid, f)
        
    # log_reg = LogReg()
    
    # log_reg.grid.fit(images.reshape((-1, np.prod(train_images.shape[1:]))), train_labels)

    # print(log_reg.grid.cv_results_)
    # error = 1 - log_reg.grid.best_score_
    # std = log_reg.grid.cv_results_['std_test_score'][log_reg.grid.best_index_]
    # additionals = {'best_params': log_reg.grid.best_params_}

    # lr = LR(C=1.)    
    # lr.fit(train_images.reshape(-1, np.prod(train_images.shape[1:])), train_labels)

    # test_images = images[train_size:]
    # test_labels = labels[train_size:]
    # print(lr.score(test_images.reshape((-1, np.prod(test_images.shape[1:]))), test_labels))

    # for i, l in zip(images[train_size:train_size + 199], labels[300:500]):

    #     pred = lr.predict(i.reshape((1, -1)))[0]
    #     if pred != l:
    #         print(pred, l)
    #         io.imshow((i/2)+0.5)
    #         io.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="python test_models.py")
    args = parser.parse_args()

    main(args)
