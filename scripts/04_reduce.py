# Load libraries
import os
import sys
import time
import datetime

from termcolor import colored, cprint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from os import listdir
from os.path import isfile, join

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
    args[3].split(':')[0] : args[3].split(':')[1]
}

chrom = dict['chrom']
epoch = int(dict['epoch'])
bottleneck = int(dict['bottle'])

path_input = glob.PATH_GENO_segmented + f"chrom_{chrom}/"

# We have to iterate over the files in shuffled and unshuffled
onlyfiles_input = [f for f in listdir(path_input) if f.split('_')[-1] != 'params']
onlyfiles_input = [f for f in onlyfiles_input if f.split('_')[-1] != 'reduced']

# iterating over folders (shuffled - unshuffled)
for folder in onlyfiles_input:
    path_output = f"{path_input}{folder}_reduced/"
    os.system(f"mkdir {path_output}")
    onlyfiles_output = [f for f in listdir(path_output) if isfile(join(path_output, f))]
    path_genos = path_input + folder + "/"
    onlyfiles_input2 = [f for f in listdir(path_genos) if isfile(join(path_genos, f))]
    onlyfiles = [x for x in onlyfiles_input2 if x not in onlyfiles_output]
    i = len(onlyfiles_output)
    for file in onlyfiles:
        
        num = file.split("_")[0]
        print(f"advancement for {folder} chrom {chrom} = {round((i/len(onlyfiles_input2))*100,2)}% now doing file nr {num}")

        # First finding the hyperparams corresponding to the file
        path_hyper_folder = path_input + folder + '_hyper_params/'
        hyper_param_file = [f for f in listdir(path_hyper_folder) if f.split('_')[0] == num ]
        hyper_param_file = [f for f in hyper_param_file if f.split('.')[-1] != 'txt' ]
        path_hyper = path_hyper_folder + hyper_param_file[0]
        params = pd.read_pickle(path_hyper)
        activation = params['activation'][0]
        drop_rate = float(params['drop_rate'][0])
        #bottleneck = int(params['n_bottleneck'][0])
        depth = int(params['depth'][0])
        
        # divide in train - test sets
        genos = (pd.read_pickle(path_genos+file) + 1)/2
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
        AE = create_autoencoder(input_len, bottleneck, drop_rate, activation, depth, optimizer = 'Adam', loss = 'binary_crossentropy', output_activation = 'sigmoid')
        history = AE.fit(X_train,
                    X_train,
                    validation_data=(X_test, X_test),
                    epochs = epoch,
                    verbose = 0)

        metric_plt = metrics_viewer(history)
        metric_plt.savefig(f"temporary_plots/metrics_dimensionnality_reduction.png")
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
        low_dim.to_pickle(f"{path_output}{file}")
        i += 1