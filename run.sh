#!/bin/bash

#SBATCH --job-name=oop
#SBATCH --time=01:50:00
#SBATCH --mem=1000G

module purge
module load SciPy-bundle/2023.02-gfbf-2022b
module load tqdm
module load scikit-learn

pip install --upgrade pip
pip install --upgrade wheel
pip install -r require.txt

python3 habrok_random_search.py