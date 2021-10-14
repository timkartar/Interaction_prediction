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

#python main.py --train_file train.txt --valid_file valid.txt -c config_30.json
#python main.py --train_file train_10.txt --valid_file valid_10.txt -c my_config.json 
#python -W ignore main.py --train_file train_5.txt --valid_file valid_5.txt -c config_dna2vec_5.json --eval_every 1
python -W ignore main.py --train_file corr_train.txt --valid_file corr_valid.txt -c config_dna2vec_5.json --eval_every 1
