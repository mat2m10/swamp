import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from termcolor import colored, cprint

from .glob import glob

def geno_segmenter_on_snp_reduced_dimension(genos, chrom, sample_size, tool, depth, epoch):
    
    # If shuffled is true (default) the sample_size chunks are shuffled
    t = time.time()
    print(colored(f"Doing dimensionality reduction on {genos.shape[1]} snps", 'blue'), colored(' [can take time, be patient]', 'red'))
    if tool == 'pca':
        from swamp.ld_clusters import ld_groups_pca
        genos, LDs = ld_groups_pca(genos, sample_size = sample_size)
    elif tool == 'ae':
        from swamp.ld_clusters import ld_groups_ae
        genos, LDs = ld_groups_ae(genos, sample_size = sample_size, depth = depth, epoch = epoch)
    else:
        print('input correct tool: pca or ae')
        
    elapsed = round(time.time() - t,2)
    cprint(f"This took {elapsed} seconds",'red')
    
    genos = genos.sort_values(by='', ascending=True)
    for num in genos.ordering.unique():
        to_save = genos.loc[genos['ordering'] == num]
        to_save = to_save.drop(columns=['ordering']).T
        size = to_save.shape[1]
        to_save.to_pickle((f"{new_output}{num}_{size}_snps_chr{chrom}.pkl"))
    
    # Message the user
    if shuffled:
        cprint(f"Splitting segmented data into chuncks of more or less {sample_size} unsimilar snps", "blue")
    else:
        cprint(f"Splitting segmented data into chuncks of similar snps", "blue")
    
    # Delete rendundent data
    os.system(f"rm -rf {new_output}plink.raw")

def AF_per_population_maker(total, humans):
    """
    Function to calculate the Allele Frequencies of different populations
    """
    total = total.T
    humans = humans[['ancestry']].join(total)
    afr = humans.loc[humans['ancestry'] == 0]
    amr = humans.loc[humans['ancestry'] == 1]
    eas = humans.loc[humans['ancestry'] == 2]
    eur = humans.loc[humans['ancestry'] == 3]
    sas = humans.loc[humans['ancestry'] == 4]
    afr = afr.drop(columns=['ancestry'])
    amr = amr.drop(columns=['ancestry'])
    eur = eur.drop(columns=['ancestry'])
    sas = sas.drop(columns=['ancestry'])
    eas = eas.drop(columns=['ancestry'])
    df = pd.DataFrame(total.columns, columns = ['snp_name'])
    total = np.where(total == 0, -0.5, total)
    minor = np.where(total == 1., 0, total)
    len_min = minor.shape[0]*-1
    df['total_af'] = minor.sum(axis = 0)/len_min
    
    total = np.where(afr == 0, -0.5, afr)
    minor = np.where(total == 1., 0, total)
    len_min = minor.shape[0]*-1
    df['afr_af'] = minor.sum(axis = 0)/len_min
    
    total = np.where(amr == 0, -0.5, amr)
    minor = np.where(total == 1., 0, total)
    len_min = minor.shape[0]*-1
    df['amr_af'] = minor.sum(axis = 0)/len_min
    
    total = np.where(eur == 0, -0.5, eur)
    minor = np.where(total == 1., 0, total)
    len_min = minor.shape[0]*-1
    df['eur_af'] = minor.sum(axis = 0)/len_min
    
    total = np.where(eas == 0, -0.5, eas)
    minor = np.where(total == 1., 0, total)
    len_min = minor.shape[0]*-1
    df['eas_af'] = minor.sum(axis = 0)/len_min
    
    total = np.where(sas == 0, -0.5, sas)
    minor = np.where(total == 1., 0, total)
    len_min = minor.shape[0]*-1
    df['sas_af'] = minor.sum(axis = 0)/len_min

    del(total)
    del(afr)
    del(amr)
    del(eur)
    del(sas)
    del(eas)
    return df
        