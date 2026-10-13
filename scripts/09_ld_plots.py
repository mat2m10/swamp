import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from os import listdir
from os.path import isfile, join
import pandas as pd
from termcolor import colored, cprint
from swamp.glob import glob
from swamp.helpers import count_AF, fa, create_plot
from swamp.segmenter import AF_per_population_maker
args = sys.argv

# Creating a dictonary of keys and variables to use as an input
dict= {
    args[1].split(':')[0] : args[1].split(':')[1], # chrom
}

chrom = int(dict['chrom'])

PATH_input = f"{glob.PATH_GENO_raw}chrom_{chrom}/"
PATH_ld = f"{glob.PATH_GENO_raw_ld}chrom_{chrom}/"
PATH_output = f"{glob.PATH_PLOTS}LD_plots/"
os.system(f"mkdir {PATH_output}chrom_{chrom}")
# If we have 1000G labeled data in all.panel (csv format)
try:
    PATH_humans = glob.PATH_PHENO_ancestry
    humans = pd.read_csv(PATH_humans, sep="\t")
    humans = humans.drop(columns=['Unnamed: 5', 'Unnamed: 6'])
    humans = humans.rename({'sample': 'FID', 'sample.1': 'IID'}, axis=1)
    humans['ancestry'] = humans['super_pop']
    humans['ancestry'].mask(humans['super_pop'] == 'AFR', 0, inplace=True)
    humans['ancestry'].mask(humans['super_pop'] == 'AMR', 1, inplace=True)
    humans['ancestry'].mask(humans['super_pop'] == 'EAS', 2, inplace=True)
    humans['ancestry'].mask(humans['super_pop'] == 'EUR', 3, inplace=True)
    humans['ancestry'].mask(humans['super_pop'] == 'SAS', 4, inplace=True)
except Exception as e:
    print("No 1000G labeled data")

onlyfiles_input = [f for f in listdir(PATH_ld) if isfile(join(PATH_ld, f))]

for file in onlyfiles_input:
    path_file = PATH_input + file
    path_ld = PATH_ld + file
    genos = pd.read_pickle(path_file)
    LDs = pd.read_pickle(path_ld)
    
    AFs = count_AF(genos)
    
    # For the total population
    hue = fa(AFs['total_af'])
    os.system(f"mkdir {PATH_output}chrom_{chrom}/{file}")
    create_plot(hue, LDs, "dim1", "dim2", f"AF differences for the whole population", PATH_output+f"chrom_{chrom}/{file}/AF_differences_total_pop.png")
    try:
        af = AF_per_population_maker(genos.T, humans)
        # For african ancestry
        hue = fa(af['afr_af'])
        create_plot(hue, LDs, "dim1", "dim2", f"AF differences in african population", PATH_output+f"chrom_{chrom}/{file}/AF_differences_afr_pop.png")

        # For american ancestry
        hue = fa(af['amr_af'])
        create_plot(hue, LDs, "dim1", "dim2", f"AF differences in american population", PATH_output+f"chrom_{chrom}/{file}/AF_differences_amr_pop.png")

        # For european ancestry
        hue = fa(af['eur_af'])
        create_plot(hue, LDs, "dim1", "dim2", f"AF differences in european population", PATH_output+f"chrom_{chrom}/{file}/AF_differences_eur_pop.png")

        # For east-asian ancestry
        hue = fa(af['eas_af'])
        create_plot(hue, LDs, "dim1", "dim2", f"AF differences in east-asian population", PATH_output+f"chrom_{chrom}/{file}/AF_differences_eas_pop.png")

        # For south-asian ancestry
        hue = fa(af['sas_af'])
        create_plot(hue, LDs, "dim1", "dim2", f"AF differences in south-asian population", PATH_output+f"chrom_{chrom}/{file}/AF_differences_sas_pop.png")
    except Exception as e:
        print(e)