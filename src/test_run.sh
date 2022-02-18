#!/bin/bash
#SBATCH --time=1440
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8GB
#SBATCH --mail-user=raktimmi@usc.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH -o ./report/output.%x.%j.out 
#SBATCH --partition=rohs
#SBATCH --gres=gpu:2

python -W ignore main.py --train_file mini_train.txt --valid_file mini_valid.txt -c combined_config.json --eval_every 1
