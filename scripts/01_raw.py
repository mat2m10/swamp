import os
import sys
args = sys.argv
from swamp.glob import glob
import numpy as np
import pandas as pd
import math
from termcolor import colored, cprint

"""
The goal of this file is to divide snps using plink in groups of #macro sample size snps having the same allele frequencies
"""
# Creating a dictonary of keys and variables to use as an input
dict= {
    args[1].split(':')[0] : args[1].split(':')[1], # name of the file
    args[2].split(':')[0] : args[2].split(':')[1], # chrom
    args[3].split(':')[0] : args[3].split(':')[1], # macro sample size
}

name = dict['name']
chrom = int(dict['chrom'])
plink_snps_pool = int(dict['macro']) # You want to select an array of '#macro' snps with the same AF

os.system(f"mkdir {glob.PATH_GENO}raw") # make directory for the output files

PATH_input = glob.PATH_GENO_full
PATH_output = glob.PATH_GENO_raw

maf = 0
max_mafs = np.arange(51)/100 # make an array of allele frequencies that we can iterate on
max_mafs = list( dict.fromkeys(max_mafs) ) # make it so that you dont have any duplicates

max_mafs = max_mafs[1:] # skip zero

from swamp.helpers import count_AF
from swamp.helpers import fa
import csv
csv.field_size_limit(sys.maxsize)
os.system(f"mkdir {PATH_output}chrom_{chrom}")
i = 0
for max_maf in max_mafs:
    # read trough all the max minor allele frequencies to create groups of snps from plink_snps_pool size
    if max_maf > maf:
        os.system(f"./plink2 --bfile {PATH_input}{name} --chr {chrom} --maf {maf} --max-maf {max_maf} --mac 50 --recode A --out ./plink")

        try:
            with open('./plink.raw') as f: # Read the plink.raw file, and count the number of snps
                reader = csv.reader(f, delimiter='\t', skipinitialspace=True)
                first_row = next(reader)
                num_cols = len(first_row) - 6 # -6 because of FID IID ...

            if num_cols > plink_snps_pool: # You want at least "plink_snps_pool"number of snps
                cprint(f"making {math.floor(num_cols/plink_snps_pool)} group(s) of {plink_snps_pool} snps with similar allele frequencies from {num_cols} snps", 'green')
                genos = pd.read_csv('./plink.raw', sep="\t") # convert to pandas dataframe
                print(genos)
                to_drop = ['FID','IID','PAT','MAT','SEX', 'PHENOTYPE']
                genos = genos.drop(to_drop, axis=1).astype(int) - 1
                AF = count_AF(genos)
                to_keep = genos.T
                to_keep['AF'] = list(AF['total_af'])
                to_keep = to_keep.sort_values(by=['AF'])
                while to_keep.shape[0] >= plink_snps_pool:

                    head = to_keep.head(plink_snps_pool)
                    min = fa(head.head(1)['AF'])[0]
                    max = fa(head.tail(1)['AF'])[0]
                    true_max = float(head.tail(1)['AF'])
                    head = head.drop(columns=['AF']).T
                    head.to_pickle(f"{PATH_output}chrom_{chrom}/{i}_maf_{min}_{max}.pkl") # save the group snps to pickle format
                    to_keep = to_keep.tail(to_keep.shape[0] - plink_snps_pool)
                    i+=1
                maf = true_max

            else:
                pass
        except Exception as e:
            pass
    else:
        pass
    os.system("rm plink.raw")

os.system("rm plink.log")
