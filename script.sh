#!/bin/bash

#SBATCH --job-name=slovotvirModel      ## Name of the job
#SBATCH --output=slovotvirOut.out    ## Output file
#SBATCH --ntasks-per-node=10
## notifications par mail
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=alexey.koshevoy@univ-amu.fr

## Load the python interpreter
module load python

## Execute the python script and pass the argument/input '90'
srun python src/model.py 1 0