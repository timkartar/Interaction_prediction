#!/bin/bash
#SBATCH --time=1440
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8GB
#SBATCH --mail-user=raktimmi@usc.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH -o ./report/output.%x.%j.out 
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100:2

python -W ignore main.py --train_file train.txt --valid_file valid.txt -c config.json --eval_every 1
