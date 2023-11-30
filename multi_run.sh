#!/bin/bash

#SBATCH --job-name=random_search
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G


module purge
module load SciPy-bundle/2023.02-gfbf-2022b
module load tqdm
module load scikit-learn

pip install --upgrade pip
pip install --upgrade wheel
pip install -r require.txt

python3 parallelized_random_search.py