import os
import sys
from os import listdir
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
from termcolor import cprint
from swamp.hypertuner import best_autoencoder

from swamp.glob import glob

# load arguments from the terminal
args = sys.argv
dict= {
    args[1].split(':')[0] : args[1].split(':')[1]
}

type = dict["type"]


path_primal = glob.PATH_GENO + 'concat/'

onlyfiles_input = [f for f in listdir(path_primal) if f.split('_')[0] != 'best']
different_files = [f.split('.')[0] for f in onlyfiles_input]
if type in different_files:
    types = [dict["type"]+'.pkl']
else:
    types = [f for f in onlyfiles_input if len(f.split('_'))>1]

path_params = glob.PATH_PARAMS_concat
for type in types:
    reduced = pd.read_pickle(path_primal + type)
    cprint(f"Finding the best hyperparams for {type}", "blue")
    result = pd.DataFrame(best_autoencoder(reduced, path_params).cv_results_).sort_values(by='rank_test_score',ignore_index=True)
    result = result.rename(columns={"param_activations": "activation",
                    "param_drop_rate": "drop_rate",
                    "param_n_bottleneck":"n_bottleneck",
                    'param_depth':"depth"})
    result.to_pickle(path_primal +f"best_params_{type.split('.')[0]}.pkl")
    result = result[['activation', "drop_rate", "n_bottleneck", "depth"]].head(1)
    
    # Actually print the best params
    cprint(f"The best hyperparams of this iteration for {type}:", "blue")
    cprint(result,'red')
    cprint(f"Stored in pandas dataframe at: {path_primal}best_params_{type.split('.')[0]}.pkl", "yellow")

    # Also save the best params in txt format to read while debugging
    with open(path_primal +f"best_params_{type.split('.')[0]}.txt", mode='w') as file_object:
        print(result, file=file_object)