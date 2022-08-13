#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=train
#SBATCH -o res.out


source /home/zhoukun/miniconda3/bin/activate new


# you can set the hparams by using --hparams=xxx


python train.py -l logdir \
-o outdir_new --n_gpus=1 -c '/home/panzexu/kun/nonparaSeq2seqVC_code-master/checkpoint_234000' --warm_start

