import os
import sys
from os import listdir
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd

from swamp.glob import glob

# load arguments from the terminal
args = sys.argv
dict= {
    args[1].split(':')[0] : args[1].split(':')[1]
}

type = dict["type"]
path_input = glob.PATH_GENO_segmented
chroms = [f for f in listdir(path_input)]
# Collect every reduced segment first and concatenate once at the end:
# concatenating inside the loop re-copies everything read so far on every
# iteration, which is quadratic in the number of segments.
reduced_segments = []
for chrom in chroms:
    path_data = [f for f in listdir(f"{path_input}{chrom}")if f.split('_')[-1] != 'reduced']
    path_data = [f for f in path_data if f.split('_')[-1] != 'params']
    if dict["type"] == "":
        types = path_data
    else: 
        types = [f"{dict['type']}"]
    print(types)
    for type in types:
        new_path = f"{path_input}{chrom}/{type}_reduced/"
        subdatas = [f for f in listdir(new_path)]
        for subdata in subdatas:
            path_subdata = f"{path_input}{chrom}/{type}_reduced/{subdata}"
            reduced_segments.append(pd.read_pickle(path_subdata)[['dim1','dim2']].T)

genos = pd.concat(reduced_segments, ignore_index=True)
del reduced_segments
genos = genos.T
name_file = '_'.join(types)
path_output = glob.PATH_GENO
os.system(f"mkdir {path_output}concat")
genos.to_pickle(f"{path_output}concat/{name_file}.pkl")