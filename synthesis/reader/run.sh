#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=train
#SBATCH -o res.out 

source /home/zhoukun/miniconda3/bin/activate s2s

python extract.py
