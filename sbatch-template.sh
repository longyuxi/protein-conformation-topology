#!/bin/bash
#SBATCH --partition=scavenger
#SBATCH --time='1:00:00'
#SBATCH --chdir=/work/yl708/protein-conformation-topology
#SBATCH --mem=2G
#SBATCH --output=%OUTPUT_FILE%
#SBATCH --error=%ERROR_FILE%

date
hostname

%COMMAND%
