#!/bin/bash


#SBATCH --gres=gpu:1
#SBATCH --job-name=inference
#SBATCH -o res_2.out 

source /home/zhoukun/miniconda3/bin/activate new

python inference.py -c '/home/panzexu/kun/nonparaSeq2seqVC_code-master/pre-train_IS_0019/outdir_new/checkpoint_44900' --num 20 --hparams validation_list='/home/panzexu/kun/nonparaSeq2seqVC_code-master/fine-tune-21/reader/emotion_list_0019/evaluation_mel_list.txt',SC_kernel_size=1
