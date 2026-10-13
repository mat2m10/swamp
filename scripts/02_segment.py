import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import datetime
from os import listdir
from os.path import isfile, join
import pandas as pd
from termcolor import colored, cprint
from swamp.glob import glob
args = sys.argv

'''
The goal of this file is to segment the macro goups in #sample_size size based on LD and similarity
'''
# Creating a dictonary of keys and variables to use as an input
dict= {
    args[1].split(':')[0] : args[1].split(':')[1], # chrom
    args[2].split(':')[0] : args[2].split(':')[1], # sample size
    args[3].split(':')[0] : args[3].split(':')[1], # tool being used to segment, either AE or PCA
    args[4].split(':')[0] : args[4].split(':')[1], # depth of the AE model if that is the tool being used
    args[5].split(':')[0] : args[5].split(':')[1], # number of epoch of the AE model if its the one chosen
}

chrom = int(dict['chrom'])
sample_size = int(dict['sample_size'])
tool = dict['tool']
depth = int(dict['depth'])
epoch = int(dict['epoch'])

PATH_input = f"{glob.PATH_GENO_raw}chrom_{chrom}/"

# Make directory for output
os.system(f"mkdir {glob.PATH_GENO}segmented")
os.system(f"mkdir {glob.PATH_GENO_segmented}chrom_{chrom}")
os.system(f"mkdir {glob.PATH_GENO_segmented}chrom_{chrom}/shuffled")
os.system(f"mkdir {glob.PATH_GENO_segmented}chrom_{chrom}/unshuffled")
os.system(f"mkdir {glob.PATH_GENO_raw_ld}chrom_{chrom}")

PATH_output = f"{glob.PATH_GENO_segmented}chrom_{chrom}"
PATH_output_ld = f"{glob.PATH_GENO_raw_ld}chrom_{chrom}/"
PATH_output_shuffled = PATH_output + "/shuffled/"
PATH_output_unshuffled = PATH_output + "/unshuffled/"

# Going through the input files and checking if the work has already be done before
onlyfiles_input = [f for f in listdir(PATH_input) if isfile(join(PATH_input, f))]
onlyfiles_output = [f for f in listdir(PATH_output_ld) if isfile(join(PATH_output_ld, f))]
onlyfiles = [x for x in onlyfiles_input if x not in onlyfiles_output]
i = len(onlyfiles_output)

for file in onlyfiles:
    
    clus_nr = file.split('.pkl')[0].split('_')[0]
    print(colored(f"advancement for chrom {chrom} = {round((i/len(onlyfiles_input))*100,2)}% now doing file nr {clus_nr}","yellow"))
    genos = pd.read_pickle(PATH_input+file)
    t = time.time()
    print(colored(f"Doing dimensionality reduction on {genos.shape[1]} snps using {tool}", 'blue'), colored(' [can take time, be patient]', 'red'))
    if tool == 'pca':
        from swamp.ld_clusters import ld_groups_pca
        genos, LDs = ld_groups_pca(genos, sample_size = sample_size)
    elif tool == 'ae':
        from swamp.ld_clusters import ld_groups_ae
        genos, LDs = ld_groups_ae(genos, sample_size = sample_size, depth = depth, epoch = epoch)
    else:
        print('input correct tool: pca or ae')
    
    elapsed = round(time.time() - t,2)
    time_string = str(datetime.timedelta(seconds=elapsed))
    time_array = time_string.split(':')
    hours = time_array[0]
    minutes = time_array[1]
    seconds = round(float(time_array[2]),2)
    
    cprint(f"This took {hours} hours, {minutes} minutes and {seconds} seconds",'red')
    
    LDs.to_pickle(f"{PATH_output_ld}{file}")
    genos_shuffled = genos.sort_values(by='ordering_shuffled', ascending=True)
    for num in genos_shuffled.ordering_shuffled.unique():
        to_save = genos_shuffled.loc[genos_shuffled['ordering_shuffled'] == num]
        to_save = to_save.drop(columns=['ordering_shuffled','ordering_unshuffled']).T # Transpose back
        size = to_save.shape[1]
        to_save.to_pickle(f"{PATH_output_shuffled}{clus_nr}_{num}_{size}_{file}")
        
    genos = genos.sort_values(by='ordering_unshuffled', ascending=True)
    for num in genos.ordering_unshuffled.unique():
        to_save = genos.loc[genos['ordering_unshuffled'] == num]
        to_save = to_save.drop(columns=['ordering_shuffled','ordering_unshuffled']).T
        size = to_save.shape[1]
        to_save.to_pickle(f"{PATH_output_unshuffled}{clus_nr}_{num}_{size}_{file}")
    
    i+=1