import os
import sys
from os import listdir
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import datetime
import pandas as pd
from termcolor import colored, cprint
import numpy as np
import pandas as pd  # Import Pandas and Numpy to create databases
import tensorflow as tf  # Import tensorflow to do tensor manipulations

from swamp.glob import glob
from swamp.helpers import metrics_viewer, population_viewer
from swamp.models import create_autoencoder
from swamp.segmenter import AF_per_population_maker

# load arguments from the terminal
args = sys.argv
dict= {
    args[1].split(':')[0] : args[1].split(':')[1],
    args[2].split(':')[0] : args[2].split(':')[1],
    args[3].split(':')[0] : args[3].split(':')[1],

}

type = dict["type"]
epoch = int(dict['epoch'])
bottleneck = int(dict['bottle'])

print([type, epoch, bottleneck])
os.system(f"mkdir {glob.PATH_GENO}mini_dim")
path_output = glob.PATH_GENO + 'mini_dim/'
path_primal = glob.PATH_GENO + 'concat/'

onlyfiles_input = [f for f in listdir(path_primal) if f.split('_')[0] != 'best']
different_files = [f.split('.')[0] for f in onlyfiles_input]
if type in different_files:
    types = [dict["type"]+'.pkl']
else:
    types = [f for f in onlyfiles_input if len(f.split('_'))>1]

for type in types:
    # First find the best hyper parameters
    path_hyper = path_primal+f"best_params_{type}"
    params = pd.read_pickle(path_hyper)
    activation = params['activation'][0]
    drop_rate = float(params['drop_rate'][0])
    #bottleneck = int(params['n_bottleneck'][0])
    depth = int(params['depth'][0])

    genos = pd.read_pickle(path_primal+type)
    print(genos.head())

            # divide in train - test sets
    X = tf.convert_to_tensor(genos, dtype=tf.float32)
    genos = genos.sample(frac=1)
    len_test = round(genos.shape[0]/10)*3
    len_train = genos.shape[0] - len_test
    X_train = genos.head(len_train)
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test = genos.tail(len_test)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

    input_len = int(tf.shape(X_train)[1])

    # Fitting the autoencoder
    t = time.time()
    print(colored(f"Doing dimensionality reduction on {genos.shape[1]} features of {genos.shape[0]} samples using an autoencoder with {epoch} epochs", 'yellow'), colored(' [can take time, be patient]', 'red'))
    AE = create_autoencoder(input_len, bottleneck, drop_rate, activation, depth, optimizer = 'Adam', loss = 'mse', output_activation = 'linear')
    history = AE.fit(X_train,
                X_train,
                validation_data=(X_test, X_test),
                epochs = epoch,
                verbose = 0)

    metric_plt = metrics_viewer(history)
    metric_plt.savefig(f"temporary_plots/metrics_final_dimensionnality_reduction_{type.split('.')[0]}.png")
    metric_plt.close()

    elapsed = round(time.time() - t,2)
    time_string = str(datetime.timedelta(seconds=elapsed))
    time_array = time_string.split(':')
    hours = time_array[0]
    minutes = time_array[1]
    seconds = round(float(time_array[2]),2)

    cprint(f"This took {hours} hours, {minutes} minutes and {seconds} seconds",'red')

    # Getting the lower dimensionality hidden representation
    hidden_repr = AE.layers[1](X).numpy()

    # depending on how many dims in the bottleneck
    cols = []
    for i2 in range(np.shape(hidden_repr)[1]):
        cols.append(f'dim{i2+1}')
    low_dim = pd.DataFrame(hidden_repr, columns = cols)
    elapsed = round(elapsed)
    low_dim.to_pickle(
        f"{path_output}reduction_of_{type.split('.')[0]}_{bottleneck}_dims_{elapsed}_seconds.pkl")
