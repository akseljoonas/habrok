#!/bin/bash

#SBATCH --job-name=5_best_models
#SBATCH --time=10:00:00
#SBATCH --mem=4G

module purge
module load SciPy-bundle/2023.02-gfbf-2022b
module load tqdm
module load scikit-learn

pip install --upgrade pip
pip install --upgrade wheel
pip install -r require.txt

python3 habrok_random_search_heur.py