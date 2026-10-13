# swamp

Autoencoder-based **dimensionality reduction for SNP genotype data**, to visualise and capture population stratification.

swamp segments genotypes by allele frequency and LD, hypertunes and trains an autoencoder per segment, concatenates the reduced segments, and reduces them once more to a low-dimensional representation that can be plotted or used as covariates.

## Install

```bash
make install        # pip install -e .
```

Python 3 with TensorFlow/Keras, pandas, NumPy and scikit-learn (see `requirements.txt`). PLINK 2 must be on your `PATH`.

## Data

```bash
export SWAMP_DATA=/path/to/your/data/   # contains genotype/ ...
```

Developed on the public **1000 Genomes** data. No genotype data is included in this repository.

## Pipeline

| Step | Make target | Script |
|---|---|---|
| 1 | `make raw name=… chrom=… macro=…` | `01_raw.py`: PLINK files to raw genotypes |
| 2 | `make segment chrom=… sample_size=… tool=ae depth=… epoch=…` | `02_segment.py`: split by allele frequency / LD |
| 3 | `make hypertune_segment chrom=…` | `03_hypertune_segment.py` |
| 4 | `make reduce chrom=… bottle=… epoch=…` | `04_reduce.py`: autoencoder per segment |
| 5 | `make concat type=…` | `05_concat_reduced.py` |
| 6 | `make hypertune_concat type=…` | `06_hypertune_concat.py` |
| 7 | `make low_dim type=… bottle=… epoch=…` | `07_lower_dim.py`: final low-dimensional representation |
| 8–9 | `make low_dim_plots`, `make ld_plots chrom=…` | Plots |

Hyperparameter grids are in `hyper_params_to_test_*.txt`. `testing_swamp.ipynb` and `evaluate_performance.ipynb` (outputs cleared) show the method in use.

## Status

Research code, shared as is. Author: Matthieu de Hemptinne.
