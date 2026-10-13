install:
	@pip install -e .

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -f */.ipynb_checkpoints
	@rm -Rf build
	@rm -Rf */__pycache__
	@rm -Rf */*.pyc

all: install clean

raw:
	@python scripts/01_raw.py name:$$name chrom:$$chrom macro:$$macro

segment:
	@python scripts/02_segment.py chrom:$$chrom sample_size:$$sample_size tool:$$tool depth:$$depth epoch:$$epoch

hypertune_segment:
	@python scripts/03_hypertune_segment.py chrom:$$chrom

reduce:
	@python scripts/04_reduce.py chrom:$$chrom bottle:$$bottle epoch:$$epoch

concat:
	@python scripts/05_concat_reduced.py type:$$type

hypertune_concat:
	@python scripts/06_hypertune_concat.py type:$$type

low_dim:
	@python scripts/07_lower_dim.py type:$$type bottle:$$bottle epoch:$$epoch

low_dim_plots:
	@python scripts/08_create_low_dim_plots.py
	
ld_plots:
	@python scripts/09_ld_plots.py chrom:$$chrom

