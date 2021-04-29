#!/usr/bin/env bash

# Load python (euler)
module load new gcc/4.8.2 python/3.7.1

# Install requirements
pip3 install --user -r requirements.txt

# Start script (euler)
bsub -n 48 -W 4:00 -R "rusage[mem=1024]" -o "baseline_grid_search.txt" "python3 -m baseline.grid_search"
bsub -n 48 -W 4:00 -R "rusage[mem=1024]" -o "abalone_grid_search.txt" "python3 -m results.abalone.grid_search"
bsub -n 48 -W 4:00 -R "rusage[mem=1024]" -o "adult_grid_search.txt" "python3 -m results.adult.grid_search"
bsub -n 48 -W 4:00 -R "rusage[mem=1024]" -o "bcw_grid_search.txt" "python3 -m results.bcw.grid_search"
bsub -n 48 -W 4:00 -R "rusage[mem=1024]" -o "questionnaires_grid_search.txt" "python3 -m results.questionnaires.grid_search"