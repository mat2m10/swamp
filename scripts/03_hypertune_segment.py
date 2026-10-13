# import libraries
import sys
import os
import pandas as pd
from termcolor import cprint

from swamp.glob import glob
from swamp.hypertuner import best_autoencoder
from os import listdir
from os.path import isfile, join

"""
The goal of this file is to hypertune a model on the samples
"""
# load arguments from the terminal
args = sys.argv
dict= {
    args[1].split(':')[0] : args[1].split(':')[1]
}
chrom = dict['chrom']
path_input = glob.PATH_GENO_segmented + f"chrom_{chrom}/"
chrom = dict['chrom']
onlyfiles_input = [f for f in listdir(path_input)]
path_params = glob.PATH_PARAMS
# We have to iterate over the files in shuffled and unshuffled
for folder in onlyfiles_input:
    path_genos = path_input + folder + "/"
    os.system(f"mkdir {path_input + folder}_hyper_params")
    path_output = path_input + folder + '_hyper_params/'
    onlyfiles_input = [f for f in listdir(path_genos) if isfile(join(path_genos, f))]
    # The files starting with the same number, belongs to the same AF family, we do want to hypertune accordingly
    nums = []
    for element in onlyfiles_input:
        nums.append(element.split('_')[0])
    from collections import Counter
    counter_object = Counter(nums)
    keys = counter_object.keys()
    for key in keys:
        file = [file for file in onlyfiles_input if file.startswith(key+'_')].pop() # randomly select one file of the AF subgroup and hypertune on this
        mafs= file.split("_maf_")[1].split('.pkl')[0].split('_')
        maf = mafs[0]
        max_maf = mafs[1]
        path_file = path_genos + file
        genos = pd.read_pickle(path_file)
        
        cprint(f"Finding the best hyperparams for chrom {chrom}, {folder}, maf = {maf}, max_maf = {max_maf}:", "blue")
        result = pd.DataFrame(best_autoencoder(genos, path_params).cv_results_).sort_values(by='rank_test_score',ignore_index=True)
        result = result.rename(columns={"param_activations": "activation",
                       "param_drop_rate": "drop_rate",
                       "param_n_bottleneck":"n_bottleneck",
                       'param_depth':"depth"})
        result.to_pickle(path_output +f"{key}_best_params_{maf}_{max_maf}.pkl")
        result = result[['activation', "drop_rate", "n_bottleneck", "depth"]].head(1)
        
        # Actually print the best params
        cprint(f"The best hyperparams of this iteration for chrom {chrom}, maf = {maf}, max_maf = {max_maf}:", "blue")
        cprint(result,'red')
        cprint(f"Stored in pandas dataframe at: {path_output}{key}_best_params_{maf}_{max_maf}.pkl", "yellow")

        # Also save the best params in txt format to read while debugging
        with open(path_output +f"{key}_best_params_{maf}_{max_maf}.txt", mode='w') as file_object:
            print(result, file=file_object)