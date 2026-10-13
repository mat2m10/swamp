# import libraries
import os
from collections import Counter
from os import listdir
from os.path import isfile, join

import pandas as pd
from termcolor import cprint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

from swamp.models import create_autoencoder

"""
Function to get the best hyperparameters of an autoencoder model
"""
def best_autoencoder(genos, path_params):
    n_inputs = [genos.shape[1]]
    params = [line.strip() for line in open(path_params, 'r')]

    try:
        
        n_bottleneck = []
        for i in params[0].split("#")[0].strip().split(','):
            n_bottleneck.append(int(i.strip()))
            
        drop_rate = []
        for i in params[1].split("#")[0].strip().split(','):
            drop_rate.append(float(i.strip()))
            
        activations = []
        for i in params[2].split("#")[0].strip().split(','):
            activations.append(i.strip())

        depth = []
        for i in params[3].split("#")[0].strip().split(','):
            depth.append(int(i.strip()))
        cprint('SUCCES Loading hyperparams to test', 'green')
        
    except Exception as e:
        print(e)
        cprint('FAILURE Loading hyperparams to test', 'red')
        # List of hyperparameters to check
        n_bottleneck = [2] # number of node in the bottleneck, best stay at 2
        drop_rate = [0.002] # drop rate to avoid overfitting
        activations = ['elu'] # activation function
        depth = [2, 3] # the depth of the model
    
    # Create dictionary of hyperparams
    param_grid = dict(n_inputs=n_inputs, n_bottleneck=n_bottleneck, drop_rate=drop_rate, activations = activations, depth = depth)


    model = KerasRegressor(build_fn = create_autoencoder, batch_size=128, epochs=15, verbose=0)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose = 1)
    result = grid.fit(genos, genos)
    cprint(f"Done gridsearching", "blue")
    return result

